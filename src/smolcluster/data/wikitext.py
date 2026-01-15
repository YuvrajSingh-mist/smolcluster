"""WikiText-2 dataset for language modeling.

This module provides the prepare_dataset function for loading and preprocessing
the WikiText-2 dataset for causal language modeling tasks.
"""

from transformers import AutoTokenizer
from datasets import load_dataset
import os
# Partition training data across workers
from smolcluster.utils.data import get_data_indices
from torch.utils.data import DataLoader

TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=TOKEN)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class ModelArgs:
    block_size = 128

def prepare_dataset(config, world_size: int, seed: int, rank: int):
    def collate_fn(batch):
        # Extract text data
        texts = batch  # batch is list of strings

     
        input_encodings = tokenizer(texts, padding='max_length', max_length=ModelArgs.block_size, truncation=True, return_tensors="pt")
      
        input_ids = input_encodings["input_ids"]
        
        # Create labels by shifting input_ids
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift right
        labels[:, -1] = tokenizer.eos_token_id  # Let the last token be end 
      
        return input_ids, labels

    # Load full datasets
    dataset_name = config.get('dataset_name', 'wikitext')
    dataset_config = config.get('dataset_config', 'wikitext-2-v1')
    
    train_dataset = load_dataset(dataset_name, dataset_config, split="train")
    train_texts = [item["text"] for item in train_dataset if item["text"].strip()]
    val_dataset = load_dataset(dataset_name, dataset_config, split="validation")
    val_texts = [item["text"] for item in val_dataset if item["text"].strip()]
    
    # Create validation loader (same for all workers)

    val_loader = DataLoader(
        val_texts, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
        # persistent_workers=True,
        # prefetch_factor=2
    )
    

    batch_indices = get_data_indices(len(train_texts), world_size, seed)
    train_data = [train_texts[i] for i in batch_indices[rank].tolist()]
    train_loader = DataLoader(
        train_data, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
        # persistent_workers=True,
        # prefetch_factor=2
    )
    
    return train_loader, val_loader, len(tokenizer), tokenizer.pad_token_id