import os

import torch
import wandb
import random
import torch.utils.cpp_extension
from datasets import load_dataset
from reward_model import GPTRewardModelProtonet
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments,SchedulerType
import time
from transformers import TrainerCallback


class Config:
    def __init__(self,
                 proto_mode,
                 frac,
                 n_prototypes,
                 k_nearest,
                 step_erase,
                 proto_init_mode=None,
                 distance_mode="EUCLIDEAN",
                 max_length=550,
                 hidden_size=4096,
                 classifier_dropout=0.2):
        self.proto_mode = proto_mode
        self.frac = frac
        self.n_prototypes = n_prototypes
        self.k_nearest = k_nearest
        self.step_erase = step_erase
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self.proto_init_mode = proto_init_mode
        self.distance_mode = distance_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(proto_mode):
    configs = {
        "PROTONET": Config(proto_mode="PROTONET",frac=0.2, n_prototypes=36, k_nearest=3, step_erase=None,proto_init_mode="random_subset",distance_mode="EUCLIDEAN", max_length=550, classifier_dropout=0.2)
    }

    return configs.get(proto_mode, Config(proto_mode="PROTONET",frac=0.05, n_prototypes=88, k_nearest=3, step_erase=5000))



def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train", fraction=1.0):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in tqdm(dataset):
        if random.random() > fraction:
            continue
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs

def create_comparison_dataset_webgpt(dataset, fraction=1.0):
    pairs = []
    for sample in tqdm(dataset):
        # check if the data is reserved
        if random.random() > fraction:
            continue

        # extract the question part as the prompt
        prompt = sample["question"]["full_text"]  # use the "question" field

        # chose the chosen and rejected according to the value of score_0
        if sample["score_0"] > 0:
            chosen_summary = sample["answer_0"]
            rejected_summary = sample["answer_1"]
        elif sample["score_0"] < 0:
            chosen_summary = sample["answer_1"]
            rejected_summary = sample["answer_0"]
        else:  # if score_0 is 0, skip this data
            continue

        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue

        pair = {
            "chosen": prompt + "\n" + chosen_summary,
            "rejected": prompt + "\n" + rejected_summary
        }
        pairs.append(pair)

    return pairs

def create_comparison_dataset_pairwise(dataset, fraction=1.0):
    pairs = []
    for sample in tqdm(dataset):
        # check if the data is reserved
        if random.random() > fraction:
            continue

        # directly use the prompt in the sample
        prompt = sample["prompt"]

        # get the chosen and rejected summaries
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]


        if chosen_summary == rejected_summary:
            continue

        # check if the length of the summaries meets the requirements
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue

        # construct the pair and add it to the list
        pair = {
            "chosen": prompt + "\n" + chosen_summary,
            "rejected": prompt + "\n" + rejected_summary
        }
        pairs.append(pair)

    return pairs

def create_comparison_dataset_hh(dataset, fraction=1.0):
    pairs = []
    for sample in tqdm(dataset):
        # check if the data is reserved
        if random.random() > fraction:
            continue


        prompt1 = sample["chosen"].split("Assistant", 1)[0]
        prompt2 = sample["rejected"].split("Assistant", 1)[0]


        if prompt1 != prompt2:
            continue

        prompt = prompt1
        chosen_summary = sample["chosen"].replace(prompt1, "").strip()
        rejected_summary = sample["rejected"].replace(prompt2, "").strip()

        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue

        pair = {
            "chosen": prompt + "\n" + chosen_summary,
            "rejected": prompt + "\n" + rejected_summary
        }
        pairs.append(pair)

    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )




class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":

    proto_mode = "PROTONET"
    config = get_config(proto_mode)


    """
    Using protonet
    """

    if proto_mode == "PROTONET":

        frac = config.frac
        n_prototypes = config.n_prototypes
        k_nearest = config.k_nearest
        step_erase = config.step_erase
        max_length = config.max_length
        hidden_size = config.hidden_size


        pass


    print(torch.version.cuda)

    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    print(cuda_home)
    wandb.init(mode="disabled")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists("rm_checkpoint"):
        os.mkdir("rm_checkpoint")

    training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=96,
        save_steps=96,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # Create the comparisons datasets
    # comparison from human feedback
    # data_path = "CarperAI/openai_summarize_comparisons"
    # train_pairs = create_comparison_dataset(data_path, "train", fraction=frac)
    # val_pairs = create_comparison_dataset(data_path, "test", fraction=frac)

    # # web gpt comparison
    # data_path = "openai/webgpt_comparisons"
    # dataset = load_dataset(data_path, split="train")
    #
    # train_test_split = dataset.train_test_split(test_size=0.5)
    # train_raw = train_test_split['train']
    # test_raw = train_test_split['test']
    # train_pairs = create_comparison_dataset_webgpt(train_raw, fraction=frac)
    # val_pairs = create_comparison_dataset_webgpt(test_raw, fraction=frac)

    # # Dahoas/synthetic-instruct-gptj-pairwise
    # data_path = "Dahoas/synthetic-instruct-gptj-pairwise"
    # dataset = load_dataset(data_path, split="train")
    # train_test_split = dataset.train_test_split(test_size=0.5)
    # train_raw = train_test_split['train']
    # test_raw = train_test_split['test']
    # train_pairs = create_comparison_dataset_pairwise(train_raw, fraction=frac)
    # val_pairs = create_comparison_dataset_pairwise(test_raw, fraction=frac)

    # Anthropic/hh-rlhf
    data_path = "Anthropic/hh-rlhf"
    train_raw = load_dataset(data_path, split="train")
    test_raw = load_dataset(data_path, split="test")
    train_pairs = create_comparison_dataset_hh(train_raw, fraction=frac)
    val_pairs = create_comparison_dataset_hh(test_raw, fraction=frac)



    # Make pairwise datasets for training
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModelProtonet("CarperAI/openai_summarize_tldr_sft",
                                   config)

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    start_time = time.time()


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()

    end_time = time.time()

    # Calculate and print the program execution time
    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time} seconds")

