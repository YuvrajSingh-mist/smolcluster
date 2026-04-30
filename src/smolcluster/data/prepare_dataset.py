"""WikiText-2 dataset for language modeling.

This module provides the prepare_dataset function for loading and preprocessing
the WikiText-2 dataset for causal language modeling tasks.
"""

import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Partition training data across workers
from smolcluster.utils.data import get_data_indices

# Try ~/.env first (works on remote nodes where CWD may be grove's temp dir),
# then fall back to CWD/.env (works on the coordinator/local runs).
load_dotenv(Path.home() / ".env", override=False)
load_dotenv(override=False)

def _build_tokenizer(config):
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
    config, world_size: int, seed: int, rank: int, batch_size: int = None
):
    tokenizer = _build_tokenizer(config)
    block_size = int(config.get("max_seq_len", 128))

    def collate_fn(batch):
        # Extract text data
        texts = batch  # batch is list of strings

        input_encodings = tokenizer(
            texts,
            padding="max_length",
            max_length=block_size,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_encodings["input_ids"]

        # Create labels by shifting input_ids
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift right
        labels[:, -1] = tokenizer.eos_token_id  # Let the last token be end

        return input_ids, labels

    # Use provided batch_size or fall back to config
    effective_batch_size = (
        batch_size if batch_size is not None else config["batch_size"]
    )

    # Load full datasets
    dataset_name = config["dataset_name"]
    dataset_config = config["dataset_config"]

    train_dataset = load_dataset(dataset_name, dataset_config, split="train")
    train_texts = [item["text"] for item in train_dataset if item["text"].strip()]
    val_dataset = load_dataset(dataset_name, dataset_config, split="validation")
    val_texts = [item["text"] for item in val_dataset if item["text"].strip()]

    # Create validation loader (same for all workers)

    val_loader = DataLoader(
        val_texts,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        # persistent_workers=True,
        # prefetch_factor=2
    )

    batch_indices = get_data_indices(len(train_texts), world_size, seed)
    train_data = [train_texts[i] for i in batch_indices[rank].tolist()]
    train_loader = DataLoader(
        train_data,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        # persistent_workers=True,
        # prefetch_factor=2
    )

    return train_loader, val_loader, len(tokenizer), tokenizer.pad_token_id
