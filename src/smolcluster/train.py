"""
Wikitext-2 Training with EDP (Elastic Distributed Parameter Server)

This script provides both server and worker entry points for distributed
GPT training using the refactored EDP functions.

Usage:
    Server: python train.py server <hostname>
    Worker: python train.py worker <rank> <hostname>
"""
import json
import time
import math
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchinfo
import wandb
from tqdm import tqdm
import yaml

from smolcluster.models.gpt import BaseTransformer
from smolcluster.data.wikitext import prepare_dataset

from smolcluster.algorithms.EDP.server import run_edp_server
from smolcluster.algorithms.EDP.worker import run_edp_worker
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device


# -----------------------------------------------------------------------------
# Configuration and Data Loading
# -----------------------------------------------------------------------------

def load_configs():
    """Load configuration files."""
    CONFIG_DIR = Path(__file__).parent / "configs"
    
    with open(CONFIG_DIR / "gpt_config.yaml") as f:
        gpt_config = yaml.safe_load(f)
    
    with open(CONFIG_DIR / "cluster_config_edp.yaml") as f:
        cluster_config = yaml.safe_load(f)
    
    return gpt_config, cluster_config


def load_data(config, world_size: int, seed: int, rank: int):
    """dataset for the given rank."""
    
    train_loader, val_loader, vocab_size, pad_token_id = prepare_dataset(config, world_size, seed, rank)
    
    return train_loader, val_loader, vocab_size, pad_token_id
   

def setup_wandb():
    """Setup Weights & Biases authentication."""
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        logger = logging.getLogger("[INIT]")
        logger.info("✅ Logged into wandb using WANDB_API_KEY")
    else:
        logger = logging.getLogger("[INIT]")
        logger.warning("⚠️  WANDB_API_KEY not set - wandb may prompt for login")




def run_server(hostname: str):
    """Run EDP server for GPT training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("[SERVER-MAIN]")
    
    setup_wandb()
    
    # Load configs
    gpt_config, cluster_config = load_configs()
    
    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    rank = 0  # Server is rank 0
    
    # Load data
    logger.info("Loading Wikitext dataset...")
    train_loader, val_loader, vocab_size, pad_token_id = load_data(gpt_config, world_size, seed, rank)
    logger.info(f"Data ready. Train size: {len(train_loader)}, Val size: {len(val_loader)}")
    
    # Create model
    model = BaseTransformer(
        vocab_size=vocab_size,
        max_seq_len=gpt_config['max_seq_len'],
        model_dim=gpt_config['model_dim'],
        num_layers=gpt_config['num_layers'],
        num_heads=gpt_config['num_heads'],
        ff_dim=gpt_config['ff_dim'],
        dropout=gpt_config['dropout'],
    )
    device = get_device()
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=gpt_config['learning_rate'],
        weight_decay=gpt_config['weight_decay']
    )
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Run server
    logger.info("Starting EDP server...")
    run_edp_server(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=gpt_config,
        cluster_config=cluster_config,
        hostname=hostname,
        device=device,
        criterion=criterion,
    )


def run_worker(worker_rank: int, hostname: str):
    """Run EDP worker for GPT training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(f"[WORKER-{worker_rank}-MAIN]")
    
    setup_wandb()
    
    # Load configs
    gpt_config, cluster_config = load_configs()
    
    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    
    # Worker rank is 0-indexed internally but 1-indexed in command-line
    local_rank = worker_rank - 1
    
    # Get server connection info
    host_ip = cluster_config["host_ip"][hostname]
    port = cluster_config["port"]
    
    # Load data
    logger.info(f"Loading {gpt_config.get('dataset_name', 'dataset')} dataset...")
    train_loader, val_loader, vocab_size, pad_token_id = load_data(gpt_config, world_size, seed, local_rank)
    logger.info(f"Data ready. Train size: {len(train_loader)}, Val size: {len(val_loader)}")
    
    # Create model
    model = BaseTransformer(
        vocab_size=vocab_size,
        max_seq_len=gpt_config['max_seq_len'],
        model_dim=gpt_config['model_dim'],
        num_layers=gpt_config['num_layers'],
        num_heads=gpt_config['num_heads'],
        ff_dim=gpt_config['ff_dim'],
        dropout=gpt_config['dropout'],
    )
    device = get_device()
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")
    
    # Print model summary
    logger.info("Model Summary:")
    summary = torchinfo.summary(
        model, 
        input_size=(gpt_config['batch_size'], gpt_config['max_seq_len']),
        device=device,
        dtypes=[torch.long]
    )
    logger.info(f"\n{summary}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=gpt_config['learning_rate'],
        weight_decay=gpt_config['weight_decay']
    )
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Run worker
    logger.info(f"Starting EDP worker {local_rank}...")
    run_edp_worker(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=gpt_config,
        cluster_config=cluster_config,
        worker_rank=local_rank,
        hostname=hostname,
        device=device,
        criterion=criterion,
        host_ip=host_ip,
        port=port,
    )


def main():
    """Main entry point for EDP distributed training."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python train.py server <hostname>")
        print("  Worker: python train.py worker <rank> <hostname>")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "server":
        if len(sys.argv) < 3:
            print("Error: Server mode requires hostname argument")
            print("Usage: python train.py server <hostname>")
            sys.exit(1)
        hostname = sys.argv[2]
        run_server(hostname)
    
    elif mode == "worker":
        if len(sys.argv) < 4:
            print("Error: Worker mode requires rank and hostname arguments")
            print("Usage: python train.py worker <rank> <hostname>")
            sys.exit(1)
        worker_rank = int(sys.argv[2])
        hostname = sys.argv[3]
        run_worker(worker_rank, hostname)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Mode must be 'server' or 'worker'")
        sys.exit(1)

    
if __name__ == "__main__":
    main()