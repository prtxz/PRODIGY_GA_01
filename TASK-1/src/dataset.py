import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer


def load_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_text_dataset(file_path):
    dataset = load_dataset("text", data_files={"train": file_path})
    return dataset


def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length"
    )

    # IMPORTANT: Add labels
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def prepare_dataset(file_path, model_name="gpt2"):
    tokenizer = load_tokenizer(model_name)
    dataset = load_text_dataset(file_path)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset, tokenizer