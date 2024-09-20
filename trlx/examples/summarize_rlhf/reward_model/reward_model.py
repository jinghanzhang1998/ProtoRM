import torch
import numpy as np
import os
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ProtoNet import init_prototypes, random_subset_init, compute_logits_proto, compute_dist_from_protos_to_examples, calculate_diversity_loss
from ProtoNet import divergence_mse_loss

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):

        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }

class GPTRewardModelProtonet(nn.Module):
    def __init__(self, model_path,config):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)


        # This part is the parameter of prototype network
        if config.proto_mode == "PROTONET":
            self.device = config.device
            self.proto_mode = config.proto_mode
            self.proto_init_in_progress = False
            self.proto_vectors = None
            self.idx_protos_to_be_initialized = None
            self.temp_proto_vector = [] #this vector is to initialize prototype in one step

            self.proto_end_index = [] #This is the total length of prompt and answer of prototype
            self.proto_prompt_index = [] #This is the total length of prompt of prototype
            self.proto_answer_index = []
            self.proto_class = [[], []]   #This is the classification of prototype
            # self.proto_mean_prompt_index = [] #This is the mean of prompt of prototype
            self.temp_proto_vector_to_mean = [[],[]] #this vector is to initialize one prototype using N different data
            self.temp_proto_end_index_to_mean = [[],[]] #this vector is the part where padding start
            self.temp_proto_answer_length = [[],[]]
            self.temp_proto_prompt_end_index_to_mean = []
            self.N = 2 # number of vectors to initialize one prototype
            self.alpha = 1e-1
            self.log_sigma_l = None
            self.pos_imp_num = 250
            self.pos_proto_num = config.n_prototypes + self.pos_imp_num
            self.valid_protos = config.n_prototypes
            self.max_length = config.max_length


            self.trigger = 0
            self.protos_valid = config.n_prototypes #this can choose a set of prototypes to be used, protos_valid<=n_prototypes
            self.distance_mode = config.distance_mode
            self.gamma = 1.0
            self.batch_counter = 0
            self.proto_distance_threshold = 0.5
            # Initialize and parameterize the prototype network
            init_prototypes(self, config)

            # This part is to record the data for SNE
            self.counter_test = 0
            self.counter_name = 0
            self.SNE_data = []
            self.SNE_class = []
            self.SNE_divergence = []
            self.SNE_pad = []


    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]


        # update the prototype
        if self.training:
            self.batch_counter += 1



        if self.proto_mode == "PROTONET":
            # if prototype is not initialized, then update the prototype
            if self.training and self.proto_init_in_progress:
                # set the parameter as const to avoid update in back ward propagation
                self.proto_vectors.requires_grad = False


                self.idx_protos_to_be_initialized, self.proto_init_in_progress = random_subset_init(
                    self,
                    proto_vectors=self.proto_vectors, features=hidden_states,
                    input_ids= input_ids,
                    idx_protos_to_be_initialized=self.idx_protos_to_be_initialized,
                    proto_init_in_progress=self.proto_init_in_progress,
                    )




            # introduce protonet only if initialization of prototype is done
            if self.proto_init_in_progress == False:
                if self.trigger == 0:
                    # tensor_temp_proto_vector = torch.stack(self.temp_proto_vector)
                    # self.proto_vectors.data = tensor_temp_proto_vector
                    for i, temp_tensor in enumerate(self.temp_proto_vector):
                        valid_length = temp_tensor.size(0)
                        self.proto_vectors.data[i, :valid_length] = temp_tensor



                    self.proto_vectors.requires_grad = True
                    # clear temp proto to save memory
                    self.temp_proto_vector.clear()

                    self.trigger = 1

                # Get the feature representation(batch_size, max_length)
                rewards_before_proto = self.v_head(hidden_states).squeeze(-1)

                chosen_end_scores = []
                rejected_end_scores = []
                end_inds = []

                # Split the inputs and rewards into two parts, chosen and rejected
                assert len(input_ids.shape) == 2
                bs = input_ids.shape[0] // 2
                chosen = input_ids[:bs]
                rejected = input_ids[bs:]
                chosen_rewards = rewards_before_proto[:bs]
                rejected_rewards = rewards_before_proto[bs:]

                loss = 0
                inference = False

                for i in range(bs):

                    if torch.all(torch.eq(chosen[i], rejected[i])).item():
                        c_inds = (chosen[i] == self.PAD_ID).nonzero()
                        c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                        chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                        inference = True
                        continue

                    # Check if there is any padding otherwise take length of sequence
                    c_inds = (chosen[i] == self.PAD_ID).nonzero()
                    c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                    r_inds = (rejected[i] == self.PAD_ID).nonzero()
                    r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
                    end_ind = max(c_ind, r_ind)
                    end_inds.append(end_ind)

                    # Retrieve first index where trajectories diverge
                    divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
                    assert divergence_ind > 0

                    # Index into the correct rewards
                    c_truncated_pad_reward = chosen_rewards[i][:end_ind]
                    r_truncated_pad_reward = rejected_rewards[i][:end_ind]


                    if self.training:
                        weight_of_protos_for_c, radii_c, proto_dist_c,truncted_prototype_c,prompt_comp = compute_dist_from_protos_to_examples(
                                                                                                c_truncated_pad_reward,
                                                                                                divergence_ind,
                                                                                                self, distance_mode=self.distance_mode,proto_class='chosen')

                        weight_of_protos_for_r, radii_r, proto_dist_r,truncted_prototype_r,prompt_comp = compute_dist_from_protos_to_examples(
                                                                                                r_truncated_pad_reward,
                                                                                                divergence_ind,
                                                                                                self, distance_mode=self.distance_mode,proto_class='rejected')
                    else:
                        weight_of_protos_for_c, radii_c, proto_dist_c, truncted_prototype_c, prompt_comp = compute_dist_from_protos_to_examples(
                                                                                                c_truncated_pad_reward,
                                                                                                divergence_ind,
                                                                                                self, distance_mode=self.distance_mode, proto_class='unknown')

                        weight_of_protos_for_r, radii_r, proto_dist_r, truncted_prototype_r, prompt_comp = compute_dist_from_protos_to_examples(
                                                                                                r_truncated_pad_reward,
                                                                                                divergence_ind,
                                                                                                self, distance_mode=self.distance_mode, proto_class='unknown')

                    weight_c = weight_of_protos_for_c.softmax(dim=1)
                    weight_r = weight_of_protos_for_r.softmax(dim=1)

                    chosen_rewards_after_proto = torch.matmul(weight_c, truncted_prototype_c.type(truncted_prototype_c.dtype)).squeeze()
                    rejected_rewards_after_proto = torch.matmul(weight_r, truncted_prototype_r.type(truncted_prototype_r.dtype)).squeeze()

                    chosen_rewards_after_proto = chosen_rewards_after_proto[prompt_comp:]
                    rejected_rewards_after_proto = rejected_rewards_after_proto[prompt_comp:]

                    # Index into the correct rewards
                    c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
                    r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

                    weight_proto = torch.exp(- torch.tensor(self.batch_counter) * 8e-4)

                    if weight_proto > 0.5:
                        weight_proto = 0.5

                    # if weight_proto < 0.2:
                    #     weight_proto = 0.2

                    # weight_proto = 0.5

                    rewards_chosen = weight_proto * chosen_rewards_after_proto + (1-weight_proto) * c_truncated_reward
                    rewards_rejected = weight_proto * rejected_rewards_after_proto + (1-weight_proto) * r_truncated_reward
                    # rewards_chosen = c_truncated_reward
                    # rewards_rejected = r_truncated_reward

                    # Append the last rewards to the list of end scores
                    chosen_end_scores.append(rewards_chosen[-1])
                    rejected_end_scores.append(rewards_rejected[-1])

                    loss += -torch.log(torch.sigmoid(rewards_chosen - rewards_rejected)).mean()


                    loss_diversity_c = calculate_diversity_loss(proto_distance_mat=proto_dist_c,truncted_prototype=truncted_prototype_c, model=self)
                    loss_diversity_r = calculate_diversity_loss(proto_distance_mat=proto_dist_r,truncted_prototype=truncted_prototype_r, model=self)
                    loss = loss+loss_diversity_c+loss_diversity_r

                loss = loss/bs

                if not inference:
                    chosen_end_scores = torch.stack(chosen_end_scores)
                    rejected_end_scores = torch.stack(rejected_end_scores)

                if inference:
                    chosen_end_scores = torch.stack(chosen_end_scores)
                    return {"chosen_end_scores": chosen_end_scores}



        #     if prototype still in progress of initialization, then skip that part
            if self.proto_init_in_progress:
                loss_diversity = 0
                rewards = self.v_head(hidden_states).squeeze(-1)

                chosen_end_scores = []
                rejected_end_scores = []

                # Split the inputs and rewards into two parts, chosen and rejected
                assert len(input_ids.shape) == 2
                bs = input_ids.shape[0] // 2
                chosen = input_ids[:bs]
                rejected = input_ids[bs:]
                chosen_rewards = rewards[:bs]
                rejected_rewards = rewards[bs:]

                loss = 0
                inference = False
                for i in range(bs):
                    if torch.all(torch.eq(chosen[i], rejected[i])).item():
                        c_inds = (chosen[i] == self.PAD_ID).nonzero()
                        c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                        chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                        inference = True
                        continue

                    # Check if there is any padding otherwise take length of sequence
                    c_inds = (chosen[i] == self.PAD_ID).nonzero()
                    c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                    r_inds = (rejected[i] == self.PAD_ID).nonzero()
                    r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
                    end_ind = max(c_ind, r_ind)

                    # Retrieve first index where trajectories diverge
                    divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
                    assert divergence_ind > 0

                    # Index into the correct rewards
                    c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
                    r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

                    # Append the last rewards to the list of end scores
                    chosen_end_scores.append(c_truncated_reward[-1])
                    rejected_end_scores.append(r_truncated_reward[-1])

                    # Compute loss based on truncated rewards (ignore padding)
                    loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
                loss = loss / bs
                loss = loss + loss_diversity
                if loss > 1000:
                    loss = loss/10

                if not inference:
                    chosen_end_scores = torch.stack(chosen_end_scores)
                    rejected_end_scores = torch.stack(rejected_end_scores)

                if inference:
                    chosen_end_scores = torch.stack(chosen_end_scores)
                    return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }
