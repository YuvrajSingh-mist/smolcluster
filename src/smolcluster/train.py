"""
Training with EDP or SyncPS for GPT

This script provides both server and worker entry points for distributed
GPT training using either EDP or SyncPS algorithms.

Usage:
    Server: python train.py server <hostname> --algorithm <edp|syncps>
    Worker: python train.py worker <rank> <hostname> --algorithm <edp|syncps>
    
Examples:
    python train.py server mini1 --algorithm edp
    python train.py worker 1 mini2 --algorithm syncps
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torchinfo
import wandb
import yaml

from smolcluster.models.gpt import BaseTransformer
from smolcluster.data.prepare_dataset import prepare_dataset

from smolcluster.algorithms.EDP.server import run_edp_server
from smolcluster.algorithms.EDP.worker import run_edp_worker
from smolcluster.algorithms.SynchronousPS.server import run_syncps_server
from smolcluster.algorithms.SynchronousPS.worker import run_syncps_worker

from smolcluster.utils.device import get_device


# -----------------------------------------------------------------------------
# Configuration and Data Loading
# -----------------------------------------------------------------------------

def load_configs(algorithm: str = "syncps"):
    """Load configuration files.
    
    Args:
        algorithm: Either 'edp' or 'syncps' to determine which cluster config to load
    """
    CONFIG_DIR = Path(__file__).parent / "configs"
    
    with open(CONFIG_DIR / "gpt_config.yaml") as f:
        gpt_config = yaml.safe_load(f)
    
    # Load appropriate cluster config based on algorithm
    config_file = f"cluster_config_{algorithm}.yaml"
    with open(CONFIG_DIR / config_file) as f:
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




def run_server(hostname: str, algorithm: str = "syncps"):
    """Run server for GPT training.
    
    Args:
        hostname: Server hostname
        algorithm: Either 'edp' or 'syncps'
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("[SERVER-MAIN]")
    
    setup_wandb()
    
    # Load configs
    gpt_config, cluster_config = load_configs(algorithm)
    
    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    rank = 0  # Server is rank 0
    
    # Load data
    logger.info(f"Loading {gpt_config.get('dataset_name', 'dataset')} dataset...")
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
    
    # Initialize W&B
    algo_name = algorithm.upper()
    wandb.init(
        project="smolcluster",
        name=f"{algo_name}-server-{hostname}_lr{gpt_config['learning_rate']}_bs{gpt_config['batch_size']}_workers{num_workers}",
        config={
            **gpt_config,
            "server_hostname": hostname,
            "num_workers": num_workers,
            "algorithm": algorithm,
        },
    )
    
    # Run server with selected algorithm
    logger.info(f"Starting {algo_name} server...")
    if algorithm == "edp":
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
    else:  # syncps
        run_syncps_server(
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
    wandb.finish()


def run_worker(worker_rank: int, hostname: str, algorithm: str = "syncps"):
    """Run worker for GPT training.
    
    Args:
        worker_rank: Worker rank (1-indexed)
        hostname: Worker hostname
        algorithm: Either 'edp' or 'syncps'
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Worker rank is 0-indexed internally but 1-indexed in command-line
    local_rank = worker_rank - 1
    logger = logging.getLogger(f"[WORKER-{local_rank}-MAIN]")
    
    setup_wandb()
    
    # Load configs
    gpt_config, cluster_config = load_configs(algorithm)
    
    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    
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
    
    # Initialize W&B
    algo_name = algorithm.upper()
    wandb.init(
        project="smolcluster",
        name=f"{algo_name}-worker-{hostname}_rank{local_rank}_lr{gpt_config['learning_rate']}_bs{gpt_config['batch_size']}",
        config={
            **gpt_config,
            "worker_rank": local_rank,
            "worker_hostname": hostname,
            "algorithm": algorithm,
        },
    )
    
    # Run worker with selected algorithm
    logger.info(f"Starting {algo_name} worker {local_rank}...")
    if algorithm == "edp":
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
    else:  # syncps
        run_syncps_worker(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
        )
    wandb.finish()


def main():
    """Main entry point for distributed training."""
    parser = argparse.ArgumentParser(description="Distributed GPT Training with EDP or SyncPS")
    parser.add_argument("mode", choices=["server", "worker"], help="Run as server or worker")
    parser.add_argument("arg1", help="Hostname (server mode) or rank (worker mode)")
    parser.add_argument("arg2", nargs="?", help="Hostname (worker mode only)")
    parser.add_argument("-a", "--algorithm", choices=["edp", "syncps"], default="syncps",
                        help="Training algorithm to use (default: syncps)")
    
    # Handle both new argparse format and legacy positional format
    if len(sys.argv) >= 2 and sys.argv[1] in ["server", "worker"]:
        # Try to parse with argparse first
        if "--algorithm" in sys.argv or "-a" in sys.argv:
            args = parser.parse_args()
            mode = args.mode
            algorithm = args.algorithm
            
            # Parse positional args based on mode
            if mode == "server":
                hostname = args.arg1
                worker_rank = None
            else:  # worker
                worker_rank = int(args.arg1)
                hostname = args.arg2
                if hostname is None:
                    print("Error: Worker mode requires both rank and hostname")
                    parser.print_help()
                    sys.exit(1)
        else:
            # Legacy format: mode hostname [rank]
            mode = sys.argv[1]
            if len(sys.argv) < 3:
                parser.print_help()
                sys.exit(1)
            
            if mode == "server":
                hostname = sys.argv[2]
                algorithm = sys.argv[3] if len(sys.argv) > 3 else "syncps"
                worker_rank = None
            else:  # worker
                if len(sys.argv) < 4:
                    parser.print_help()
                    sys.exit(1)
                worker_rank = int(sys.argv[2])
                hostname = sys.argv[3]
                algorithm = sys.argv[4] if len(sys.argv) > 4 else "syncps"
    else:
        parser.print_help()
        sys.exit(1)
    
    # Validate algorithm
    if algorithm not in ["edp", "syncps"]:
        print(f"Error: Invalid algorithm '{algorithm}'. Must be 'edp' or 'syncps'")
        sys.exit(1)
    
    # Run appropriate mode
    if mode == "server":
        run_server(hostname, algorithm)
    elif mode == "worker":
        if worker_rank is None:
            print("Error: Worker mode requires rank argument")
            parser.print_help()
            sys.exit(1)
        run_worker(worker_rank, hostname, algorithm)

    
if __name__ == "__main__":
    main()