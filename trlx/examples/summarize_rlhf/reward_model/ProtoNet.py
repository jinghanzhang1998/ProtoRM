import torch
import torch.nn as nn
import random
import os
import numpy as np
import torch.nn.functional as F





def init_prototypes(model: nn.Module,
        config):
    # set_lambda_and_task_name_for_bert_models(model, config)
    set_prototypes_for_bert_models(model, config)
    init_logits_proto_and_sigma_for_bert_models(model, config)

def set_prototypes_for_bert_models(model,
        config):
    model.proto_init_in_progress = False
    proto_dtype = torch.float32  # torch.float16 if config.fp16 else torch.float32
    # The prototype size is (n_prototypes+pos_imp, max_length)
    prototypes = torch.rand((config.n_prototypes+model.pos_imp_num,config.max_length), requires_grad=True, dtype=proto_dtype).to(
        model.device)
    nn.init.xavier_uniform_(prototypes)
    model.proto_vectors = nn.Parameter(prototypes, requires_grad=True).to(model.device)
    print(f"Prototype shape: {model.proto_vectors.shape}")

    model.idx_protos_to_be_initialized = set(range(config.n_prototypes))
    model.proto_init_in_progress = (config.proto_init_mode == "random_subset")


# Set the initial parameters of imp
def init_logits_proto_and_sigma_for_bert_models(model,
        config):
    """
    model: transformer-based models
    """

    init_sigma_l = 5


    """For learning cluster radii"""
    with torch.no_grad():
        log_sigma_l = torch.log(torch.FloatTensor([init_sigma_l])).to(model.device)
        model.log_sigma_l = nn.Parameter(log_sigma_l, requires_grad=True).to(model.device)


# Pooling method for compressing vectors
def mean_pooling(tensor, num_segments):

    try:
        step = len(tensor) / num_segments
    except ZeroDivisionError:

        raise ValueError(f"Division by zero error: attempted to divide by {num_segments}, please check if this value is zero.")

    pooled = []

    for i in range(num_segments):
        # Calculate the start and end indices for the current segment
        start = int(i * step)
        end = int((i + 1) * step) if i != num_segments - 1 else len(tensor)

        # Calculate the mean of the segment
        segment_mean = torch.mean(tensor[start:end])
        pooled.append(segment_mean)

    return torch.stack(pooled)


# Pooling method for compressing vectors
def mean_pooling_2d(tensor, num_segments):
    N, length = tensor.shape
    step = length // num_segments

    last_segment_length = length - (step * (num_segments - 1))

    tensor_reshaped = tensor[:, :step * num_segments].reshape(N, num_segments, step)

    # Calculate the mean of each segment
    pooled = torch.mean(tensor_reshaped, dim=2)

    # If the last segment is not the same length as the others, calculate the mean of the last segment
    if last_segment_length != step:
        last_segment = tensor[:, -last_segment_length:]
        last_segment_mean = torch.mean(last_segment, dim=1, keepdim=True)
        pooled[:, -1] = last_segment_mean.squeeze()

    return pooled


def random_subset_init(
        model,
        proto_vectors: torch.Tensor,
        features: torch.Tensor,
        input_ids: torch.Tensor,
        idx_protos_to_be_initialized: set,
        proto_init_in_progress: bool,
        batch: int = 2):
    assert proto_init_in_progress



    assert len(input_ids.shape) == 2
    bs = input_ids.shape[0] // 2

    chosen_id_set = input_ids[:bs]
    rejected_id_set = input_ids[bs:]

    chosen_fea_set = features[:bs]
    rejected_fea_set = features[bs:]


    for i in range(bs):


        chosen_hidden_states = chosen_fea_set[i]  # Choose the first slice, shape is (550, 4096)
        rejected_hidden_states = rejected_fea_set[i]  # Choose the second slice, shape is (550, 4096)

        detached_chosen = model.v_head(chosen_hidden_states).squeeze(-1).detach()
        detached_rejected = model.v_head(rejected_hidden_states).squeeze(-1).detach()


        # |input_ids| = (max_length,)
        chosen_ids = chosen_id_set[i]
        rejected_ids = rejected_id_set[i]

        if len(model.temp_proto_vector_to_mean[0]) < model.N:

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_ids == model.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_ids.shape[0]
            r_inds = (rejected_ids == model.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_ids.shape[0]

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_ids != rejected_ids).nonzero()[0]
            assert divergence_ind > 0

            # truncated vector to avoid redundency

            c_truncated_reward = detached_chosen[:c_ind]
            r_truncated_reward = detached_rejected[:r_ind]


            # If the number of elements is less than self.N, continue to append elements, positive and negative examples are placed separately
            model.temp_proto_vector_to_mean[0].append(c_truncated_reward)
            model.temp_proto_vector_to_mean[1].append(r_truncated_reward)


            model.temp_proto_end_index_to_mean[0].append(c_ind)
            model.temp_proto_end_index_to_mean[1].append(r_ind)
            model.temp_proto_prompt_end_index_to_mean.append(divergence_ind)

            # Record the length of the answer
            model.temp_proto_answer_length[0].append(c_ind-divergence_ind.item())
            model.temp_proto_answer_length[1].append(r_ind-divergence_ind.item())

        if len(model.temp_proto_vector_to_mean[0]) >= model.N:

            c_max_index = max(model.temp_proto_end_index_to_mean[0])
            r_max_index = max(model.temp_proto_end_index_to_mean[1])
            c_max_answer = max(model.temp_proto_answer_length[0])
            r_max_answer = max(model.temp_proto_answer_length[1])

            # Need to take the maximum of the prompt part of the tensor to be averaged, if it exceeds the maximum sentence length, it is equal to the maximum sentence length minus the answer length
            diver_max_index = max(tensor.item() for tensor in model.temp_proto_prompt_end_index_to_mean)

            if diver_max_index + c_max_answer > model.max_length or diver_max_index + r_max_answer > model.max_length:
                diver_max_index = model.max_length - max(c_max_answer, r_max_answer)



            # Initialize the list of tensors to be averaged
            padded_tensors_chosen = []
            padded_tensors_rejected = []


            # Separate the prompt+answer combination to be averaged, then fill in 0 separately, and then combine to ensure maximum information retention
            for tensor, diver_index in zip(model.temp_proto_vector_to_mean[0], model.temp_proto_prompt_end_index_to_mean):
                prompt = tensor[:diver_index]
                answer = tensor[diver_index:]

                # For prompt, if it exceeds the truncation, fill in 0, if it is short, fill in 0
                if prompt.size(0) > diver_max_index:
                    resize_prompt = prompt[:diver_max_index]
                else:
                    pad_prompt_size = diver_max_index-prompt.size(0)
                    resize_prompt = F.pad(prompt, (0, pad_prompt_size), 'constant', 0)


                pad_answer_size = c_max_answer - answer.size(0)
                # Padding tensor
                padded_answer = F.pad(answer, (0, pad_answer_size), 'constant', 0)
                padded_tensor = torch.cat((resize_prompt, padded_answer), dim=0)
                padded_tensors_chosen.append(padded_tensor)

            for tensor, diver_index in zip(model.temp_proto_vector_to_mean[1], model.temp_proto_prompt_end_index_to_mean):
                prompt = tensor[:diver_index]
                answer = tensor[diver_index:]

                # For prompt, if it exceeds the truncation, fill in 0, if it is short, fill in 0
                if prompt.size(0) > diver_max_index:
                    resize_prompt = prompt[:diver_max_index]
                else:
                    pad_prompt_size = diver_max_index-prompt.size(0)
                    resize_prompt = F.pad(prompt, (0, pad_prompt_size), 'constant', 0)


                pad_answer_size = r_max_answer - answer.size(0)

                padded_answer = F.pad(answer, (0, pad_answer_size), 'constant', 0)
                padded_tensor = torch.cat((resize_prompt, padded_answer), dim=0)
                padded_tensors_rejected.append(padded_tensor)


            # Calculate the mean of all tensors in the list
            prototype_chosen_raw = torch.mean(torch.stack(padded_tensors_chosen), dim=0)
            prototype_rejected_raw = torch.mean(torch.stack(padded_tensors_rejected), dim=0)

            # Add the calculated mean tensor to the temp_proto_vector list
            model.temp_proto_vector.append(prototype_chosen_raw)
            model.temp_proto_vector.append(prototype_rejected_raw)

            # Add the calculated mean tensor to the proto_vectors list
            model.proto_end_index.append(prototype_chosen_raw.size(0))
            model.proto_end_index.append(prototype_rejected_raw.size(0))
            model.proto_prompt_index.append(diver_max_index)
            model.proto_prompt_index.append(diver_max_index)

            if model.idx_protos_to_be_initialized:

                first_element = next(iter(model.idx_protos_to_be_initialized))

                # Remove the first element from idx_protos_to_be_initialized
                model.idx_protos_to_be_initialized.remove(first_element)

                # The first element belongs to the first class, and the number is placed in class 0
                model.proto_class[0].append(first_element)  # chosen组成的原型计算类时是类0

                second_element = next(iter(model.idx_protos_to_be_initialized))

                # Remove the second element from idx_protos_to_be_initialized
                model.idx_protos_to_be_initialized.remove(second_element)

                # The second element belongs to the second class, and the number is placed in class 1
                model.proto_class[1].append(second_element)   # Chosen prototypes are class 0 when calculating classes

                print(f"Initializing: {first_element} {second_element}")

            # Clear the list of tensors to be averaged
            model.temp_proto_vector_to_mean[0].clear()
            model.temp_proto_vector_to_mean[1].clear()
            model.temp_proto_end_index_to_mean[0].clear()
            model.temp_proto_end_index_to_mean[1].clear()
            model.temp_proto_prompt_end_index_to_mean.clear()
            model.temp_proto_answer_length[0].clear()
            model.temp_proto_answer_length[1].clear()



        if len(idx_protos_to_be_initialized) == 0:
            print("Initializing prototypes: complete!")
            model.proto_answer_index = [end_index - prompt_index for end_index, prompt_index in zip(model.proto_end_index, model.proto_prompt_index)]
            proto_init_in_progress = False
            break
    return idx_protos_to_be_initialized, proto_init_in_progress

def compute_logits_proto(model):
    if model.proto_mode == "PROTONET":
        logits_proto = model.v_head(model.dropout(model.proto_vectors[:model.protos_valid])).squeeze(-1)
    else:
        raise NotImplementedError
    return logits_proto

def compute_dist_from_protos_to_examples(pooled_output,
        divergence_ind,
        model,
        distance_mode=None,
        proto_class = 'unknown',
        device='cuda'):
    radii = None
    if distance_mode is None:
        distance_mode = distance_mode
    # (B, N)
    weight_of_protos_for_each_ex = None
    if distance_mode == "EUCLIDEAN":
        """
        The most straightforward way: calculating Euclidean distances as in ProtoNet
        """
        proto_dist,truncted_prototype,prompt_comp = compute_proto_dist(pooled_output.to(device),divergence_ind, model, proto_class)
        weight_of_protos_for_each_ex = (- model.gamma * proto_dist)

    else:
        raise

    return weight_of_protos_for_each_ex, radii, proto_dist, truncted_prototype,prompt_comp

def process_prototypes(prototypes, prompt_indices, end_indices, prompt_comp, answer_length):
    resized_prototypes = []

    for prototype, prompt_index, end_index in zip(prototypes, prompt_indices, end_indices):
        prompt_prototype = prototype[:prompt_index]
        answer_prototype = prototype[prompt_index:end_index]

        if prompt_index >= prompt_comp:
            # Compress the prompt
            compressed_prompt = mean_pooling(prompt_prototype, prompt_comp)
        else:
            pad_prompt_size = prompt_comp - prompt_index
            compressed_prompt = F.pad(prompt_prototype, (0, pad_prompt_size), 'constant', 0)

        # Pad the answer
        pad_answer_size = answer_length - answer_prototype.size(0)
        padded_answer = F.pad(answer_prototype, (0, pad_answer_size), 'constant', 0)

        # Concatenate the compressed prompt and the padded answer
        resized_prototype = torch.cat((compressed_prompt, padded_answer), dim=0)
        resized_prototypes.append(resized_prototype)

    return torch.stack(resized_prototypes, dim=0).to(torch.float32)

def compute_proto_dist(pooled_output,
        divergence_ind,
        model,
        proto_class,
        device='cuda'):



    pooled_output_prompt = pooled_output[:divergence_ind]
    pooled_output_answer = pooled_output[divergence_ind:]
    pooled_output_answer_size = pooled_output_answer.shape[0]
    pooled_output_prompt_size = pooled_output_prompt.shape[0]
    # If the length of prompt is greater than that of answer, compress the length of prompt to the length of answer, otherwise prompt_comp is the length of prompt

    if pooled_output_prompt_size > pooled_output_answer_size:
        prompt_comp = int(pooled_output_answer_size // 2)
    else:
        prompt_comp = pooled_output_prompt_size


    if prompt_comp == 0:
        prompt_comp = 1
    if pooled_output_answer_size == 0:
        raise ValueError("pooled_output_answer_size为0,divergence_ind 的值为: {}".format(divergence_ind))
    if prompt_comp == 0:
        raise ValueError("prompt_comp为0,divergence_ind 的值为: {}".format(divergence_ind))

    pooled_output_valid_size = prompt_comp+pooled_output_answer_size

    # Compress the effective length of prompt and answer in the prototype to the effective length of pooled_output
    valid_prototypes = model.proto_vectors[:model.valid_protos]

    # Dropout
    perm_indices_class0 = torch.randperm(len(model.proto_class[0]))
    perm_indices_class1 = torch.randperm(len(model.proto_class[1]))

    num_valid_proto_class0 = int(0.9 * len(model.proto_class[0]))
    num_valid_proto_class1 = int(0.9 * len(model.proto_class[1]))

    selected_indices_class0 = perm_indices_class0[:num_valid_proto_class0]
    selected_indices_class1 = perm_indices_class1[:num_valid_proto_class1]

    valid_proto_class0 = valid_prototypes[model.proto_class[0]][selected_indices_class0]
    valid_proto_class1 = valid_prototypes[model.proto_class[1]][selected_indices_class1]


    proto_answer_length = max(model.proto_answer_index)
    answer_length = max(proto_answer_length,(pooled_output_answer.size(0)))

    proto_prompt_index_class0 = [model.proto_prompt_index[index] for index in model.proto_class[0]]
    proto_prompt_index_class1 = [model.proto_prompt_index[index] for index in model.proto_class[1]]

    proto_end_index_class0 = [model.proto_end_index[index] for index in model.proto_class[0]]
    proto_end_index_class1 = [model.proto_end_index[index] for index in model.proto_class[1]]

    # Deal with two batches of prototypes
    truncated_proto_class0 = process_prototypes(valid_proto_class0, proto_prompt_index_class0, proto_end_index_class0,
                                                prompt_comp, answer_length)
    truncated_proto_class1 = process_prototypes(valid_proto_class1, proto_prompt_index_class1, proto_end_index_class1,
                                                prompt_comp, answer_length)

    # Compress the pooled_output
    com_pooled_prompt = mean_pooling(pooled_output_prompt, prompt_comp)



    # Padding the pooled_output_answer
    pad_answer_size = answer_length - pooled_output_answer.size(0)
    padded_pooled_answer = F.pad(pooled_output_answer, (0, pad_answer_size), 'constant', 0)
    com_pooled_output = torch.cat((com_pooled_prompt, padded_pooled_answer), dim=0)

    if proto_class != 'unknown':
        if proto_class == 'chosen':
            truncted_prototype = truncated_proto_class0
        elif proto_class == 'rejected':
            truncted_prototype = truncated_proto_class1

        pooled_output_2d = com_pooled_output.unsqueeze(0).to(torch.float32)

        proto_dist = torch.cdist(pooled_output_2d.to(device), truncted_prototype.to(device))

        proto_dist = proto_dist.to(torch.float16)
        com_prototype = truncted_prototype[:, :pooled_output_valid_size].to(torch.float16)


    elif proto_class == 'unknown':

        pooled_output_2d = com_pooled_output.unsqueeze(0).to(torch.float32)

        proto_dist_class0 = torch.cdist(pooled_output_2d.to(device), truncated_proto_class0.to(device))
        proto_dist_class1 = torch.cdist(pooled_output_2d.to(device), truncated_proto_class1.to(device))

        distances = torch.cat((proto_dist_class0, proto_dist_class1), dim=1)
        labels = torch.cat((torch.zeros(proto_dist_class0.size(1)), torch.ones(proto_dist_class1.size(1))))
        labels = labels.to(device)
        num_k = 19

        # If num_k is even, add 1 to make it odd
        if num_k % 2 == 0:
            num_k = num_k + 1

        # Get the indices of the 19 nearest prototypes
        _, indices = torch.topk(distances, num_k, largest=False)

        # Calculate the number of prototypes in each class
        class0_count = torch.sum(labels[indices] == 0).item()
        class1_count = torch.sum(labels[indices] == 1).item()

        if class0_count > class1_count:
            truncted_prototype = truncated_proto_class0
            proto_dist = proto_dist_class0
        else:
            truncted_prototype = truncated_proto_class1
            proto_dist = proto_dist_class1

        proto_dist = proto_dist.to(torch.float16)
        com_prototype = truncted_prototype[:, :pooled_output_valid_size].to(torch.float16)


    # imp
    if model.training:
        lamda = estimate_lambda(model=model, proto_vectors=com_prototype,
                                semi_supervised=False)

        if model.valid_protos < model.pos_proto_num and proto_dist.min() > lamda:
            valid_length = pooled_output.size(0)  # Get the length of the current output
            model.proto_vectors.data[model.valid_protos, :valid_length] = pooled_output
            model.proto_prompt_index.append(divergence_ind.item())
            model.proto_end_index.append(valid_length)
            model.proto_answer_index.append((valid_length-divergence_ind).item())
            # Add the category of the current prototype to the category list
            if proto_class == 'chosen':
                model.proto_class[0].append(model.valid_protos)
            elif proto_class == 'rejected':
                model.proto_class[1].append(model.valid_protos)
            elif proto_class == 'unknown':
                if class0_count > class1_count:
                    model.proto_class[0].append(model.valid_protos)
                else:
                    model.proto_class[1].append(model.valid_protos)

            model.valid_protos = model.valid_protos+1
            print("Added {} prototype, current total number of prototypes is {}".format(1, model.valid_protos))

    return proto_dist,com_prototype,prompt_comp

def estimate_lambda(model,
        proto_vectors,
        semi_supervised=False):
    # estimate lambda by mean of shared sigmas
    rho = proto_vectors.var(dim=0) # (768, )
    rho = rho.mean()
    if semi_supervised:
        """Not implemented"""
        sigma = (torch.exp(model.log_sigma_l).data[0] + torch.exp(model.log_sigma_u).data[0]) / 2.

    else:
        sigma = torch.exp(model.log_sigma_l).data[0]

    alpha = torch.tensor(model.alpha)

    # lamda = -2*sigma*np.log(config.alpha) + config.dim*sigma*np.log(1+rho/sigma)
    d = proto_vectors.size(1)



    lamda = -2 * sigma * torch.log(alpha) + d * sigma * torch.log(1 + rho / sigma)


    return lamda

def divergence_mse_loss(c_reward, r_reward):
    """
    Calculates the "inverse" MSE loss between c_reward and r_reward, aiming to maximize the distance between them.

    Parameters:
    c_reward (torch.Tensor): The first tensor.
    r_reward (torch.Tensor): The second tensor.

    Returns:
    torch.Tensor: The computed loss value.
    """
    mse_loss = F.mse_loss(c_reward.float(), r_reward.float())
    divergence_loss = torch.exp(-mse_loss)
    return divergence_loss

def calculate_diversity_loss(proto_distance_mat: torch.Tensor,
        truncted_prototype: torch.Tensor,
        model=None):

    truncted_proto_vectors = truncted_prototype.to(torch.float32)
    pairwise_distances_sq = torch.cdist(truncted_proto_vectors,
                                        truncted_proto_vectors, p=2).to(dtype=torch.float16)

    threshold = model.proto_distance_threshold
    loss_mat = torch.triu(torch.clamp(threshold - pairwise_distances_sq, min=0), diagonal=1)
    loss_diversity = (loss_mat).sum()


    nonzero_elements = torch.nonzero(loss_mat).size(0)
    loss_diversity = loss_diversity / max(nonzero_elements, 1)

    return loss_diversity



def get_radii(model):
    config = model.config
    li = [1.] * config.valid_protos + [0.] * (config.n_protos - config.valid_protos)

    radii = torch.tensor(li, device=config.device) * torch.exp(model.log_sigma_l)
    return radii

def compute_logits_radii(cluster_centers,
        data,
        radii,
        prior_weight=1.):
    """Computes the logits of being in one cluster, squared Euclidean.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        log_prob: [B, N, K] logits., N = num datapoints; K = num classes
    """

    dim = data.size()[-1]
    # neg_dist = -torch.sum((data - cluster_centers) ** 2, dim=3)  # [B, N, K]
    pos_dist = torch.cdist(data, cluster_centers.to(data.dtype)) ** 2
    neg_dist = - pos_dist  # [B, N]

    logits = neg_dist / 2.0 / (radii)  # (1, 10, 7)


    norm_constant = 0.5 * dim * (torch.log(radii) + np.log(2 * np.pi))  # (1, 1, 7)
    # (1, 30, 6)
    logits = logits - norm_constant
    return logits, pos_dist

def pairwise_cos_sim(a: torch.Tensor,
        b: torch.Tensor):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res




