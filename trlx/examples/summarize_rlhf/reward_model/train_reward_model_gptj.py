import os

import torch
import wandb
import random
import time
import torch.utils.cpp_extension
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments


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
        if random.random() > fraction:
            continue

        prompt = sample["question"]["full_text"]


        if sample["score_0"] > 0:
            chosen_summary = sample["answer_0"]
            rejected_summary = sample["answer_1"]
        elif sample["score_0"] < 0:
            chosen_summary = sample["answer_1"]
            rejected_summary = sample["answer_0"]
        else:
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
        # Check if the data should be retained
        if random.random() > fraction:
            continue

        # Directly use the prompt from the sample
        prompt = sample["prompt"]

        # Get the chosen and rejected summaries
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]

        # Skip if the chosen and rejected summaries are the same
        if chosen_summary == rejected_summary:
            continue

        # Check if the summary lengths meet the requirement
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue

        # Construct the pair and add to the list
        pair = {
            "chosen": prompt + "\n" + chosen_summary,
            "rejected": prompt + "\n" + rejected_summary
        }
        pairs.append(pair)

    return pairs

def create_comparison_dataset_hh(dataset, fraction=1.0):
    pairs = []
    for sample in tqdm(dataset):
        # Check if the data should be retained
        if random.random() > fraction:
            continue

        # Extract the content before "Assistant"
        prompt1 = sample["chosen"].split("Assistant", 1)[0]
        prompt2 = sample["rejected"].split("Assistant", 1)[0]

        # Skip if prompt1 is not equal to prompt2
        if prompt1 != prompt2:
            continue

        # If prompt1 is equal to prompt2, update prompt and summaries
        prompt = prompt1
        chosen_summary = sample["chosen"].replace(prompt1, "").strip()
        rejected_summary = sample["rejected"].replace(prompt2, "").strip()

        # Check if the summary lengths meet the requirement
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue

        # Construct the pair and add to the list
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        eval_steps=192,
        save_steps=96,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft")

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    frac = 0.2

    # # Create the comparisons datasets
    # data_path = "CarperAI/openai_summarize_comparisons"
    # train_pairs = create_comparison_dataset(data_path, "train", fraction=frac)
    # val_pairs = create_comparison_dataset(data_path, "test", fraction=frac)

    # web gpt comparison
    # data_path = "openai/webgpt_comparisons"
    # dataset = load_dataset(data_path, split="train")
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
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    start_time = time.time()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()

    # Record the end time
    end_time = time.time()

    # Calculate and print the program's execution time
    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time} seconds")

