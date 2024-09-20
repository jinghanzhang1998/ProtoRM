import os

import torch
import wandb
from datasets import load_dataset
import torch.utils.cpp_extension
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer



# def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train"):
#     dataset = load_dataset(path, split=split)
#     pairs = []
#     for sample in tqdm(dataset):
#         pair = {}
#         prompt = sample["prompt"]
#         chosen_summary = sample["chosen"]
#         rejected_summary = sample["rejected"]
#         if chosen_summary == rejected_summary:
#             continue
#         if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
#             continue
#         pair["chosen"] = prompt + "\n" + chosen_summary
#         pair["rejected"] = prompt + "\n" + rejected_summary
#         pairs.append(pair)
#     return pairs

def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train", fraction=1.0):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in tqdm(dataset):
        if random.random() > fraction:  # 只保留一定比例的数据
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
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data), dtype=torch.float)
        return batch


def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    label_ids = eval_preds.label_ids

    # 确保 predictions 是 NumPy 数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()

    # 如果predictions和label_ids的长度是奇数，则移除最后一个元素
    if len(predictions) % 2 != 0:
        predictions = predictions[:-1]
    if len(label_ids) % 2 != 0:
        label_ids = label_ids[:-1]

    # 将predictions分成score_1和score_2
    score_1 = predictions[::2]  # 偶数索引
    score_2 = predictions[1::2]  # 奇数索引

    # 将label_ids分成labels
    labels = label_ids[::2]  # 偶数索引的label_ids

    # 计算预测标签
    predicted_labels = np.where(score_1 > score_2, 0, 1)

    # 处理score_1等于score_2的情况
    uncertain = score_1 == score_2
    correct_predictions = (predicted_labels == labels) & ~uncertain

    # 计算准确率（不确定情况不计入）
    acc = np.mean(correct_predictions[~uncertain])

    return {"accuracy": acc}


if __name__ == "__main__":

    print(torch.version.cuda)

    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    print(cuda_home)
    wandb.init(mode="disabled")

    reward_name = 'microsoft/deberta-v3-large'
    tokenizer = DebertaV2Tokenizer.from_pretrained(reward_name)
    model = DebertaV2ForSequenceClassification.from_pretrained(reward_name, num_labels=1)
    model.to('cuda')

    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists("rm_checkpoint"):
        os.mkdir("rm_checkpoint")

    training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # # Initialize the reward model from the (supervised) fine-tuned GPT-J
    # model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft")

    # Freeze the first 70% of the hidden layers of the reward model backbone
    # layers = model.transformer.h
    # num_layers = len(layers)
    # num_unfrozen = int(0.3 * num_layers)
    # for layer in layers[:-num_unfrozen]:
    #     layer.requires_grad_(False)
    encoder_layers = model.deberta.encoder.layer
    num_layers = len(encoder_layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in encoder_layers[:-num_unfrozen]:
        for param in layer.parameters():
            param.requires_grad = False

    # Create the comparisons datasets
    data_path = "CarperAI/openai_summarize_comparisons"

    # Make pairwise datasets for validation
    max_length = 550

    val_pairs = create_comparison_dataset(data_path, "test",fraction=1)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()


    fractions = [0.1, 0.2, 0.3]  # 不同的数据集比例
    for frac in fractions:
        print(f"Training with {frac * 100}% of the training data")
        train_pairs = create_comparison_dataset(data_path, "train", fraction=frac)
        train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        trainer.train()

        # 将准确率结果输出到文本文件中
        result = trainer.evaluate()
        with open(f"accuracy_{int(frac * 10)}.txt", "w") as file:
            file.write(f"Accuracy with {frac * 100}% training data: {result['eval_accuracy']}\n")



    # train_pairs = create_comparison_dataset(data_path, "train")
    # val_pairs = create_comparison_dataset(data_path, "test")
    #
    # # Make pairwise datasets for training
    # max_length = 550
    # train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    # val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)
    #
    # # Create the collator to gather batches of pairwise comparisons
    # data_collator = DataCollatorReward()
    #
    # Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     compute_metrics=compute_metrics,
    #     eval_dataset=val_dataset,
    #     data_collator=data_collator,
    # ).train()
