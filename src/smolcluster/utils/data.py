"""Dataset splitting utilities — partitions indices uniquely across workers for distributed data loading."""
# For spliting the data uniquely across workers
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Try ~/.env first (works on remote nodes where CWD may be grove's temp dir),
# then fall back to CWD/.env (works on the coordinator/local runs).
load_dotenv(Path.home() / ".env", override=False)
load_dotenv(override=False)


def get_data_indices(dataset_length: int, world_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(dataset_length, generator=generator)
    split_indices = torch.chunk(indices, world_size)
    return split_indices


def _build_tokenizer(config: dict) -> Any:
    tokenizer_name = config.get("tokenizer", "openai-community/gpt2")
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def prepare_dataset(
    config: dict, world_size: int, seed: int, rank: int, batch_size: int = None
):
    tokenizer = _build_tokenizer(config)
    block_size = int(config.get("max_seq_len", 128))

    def collate_fn(batch):
        texts = batch  # batch is list of strings
        input_encodings = tokenizer(
            texts,
            padding="max_length",
            max_length=block_size,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = input_encodings["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift right
        labels[:, -1] = tokenizer.eos_token_id  # Let the last token be end
        return input_ids, labels

    effective_batch_size = (
        batch_size if batch_size is not None else config["batch_size"]
    )

    dataset_name = config["dataset_name"]
    dataset_config = config["dataset_config"]

    train_dataset = load_dataset(dataset_name, dataset_config, split="train")
    train_texts = [item["text"] for item in train_dataset if item["text"].strip()]
    val_dataset = load_dataset(dataset_name, dataset_config, split="validation")
    val_texts = [item["text"] for item in val_dataset if item["text"].strip()]

    val_loader = DataLoader(
        val_texts,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    batch_indices = get_data_indices(len(train_texts), world_size, seed)
    train_data = [train_texts[i] for i in batch_indices[rank].tolist()]
    train_loader = DataLoader(
        train_data,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return train_loader, val_loader, len(tokenizer), tokenizer.pad_token_id
