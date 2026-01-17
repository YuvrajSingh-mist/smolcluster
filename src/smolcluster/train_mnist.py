#!/usr/bin/env python3
"""
MNIST Training with EDP or SyncPS

This script provides both server and worker entry points for distributed
MNIST training using either EDP or SyncPS algorithms.

Usage:
    Server: python train_mnist.py server <hostname> --algorithm <edp|syncps>
    Worker: python train_mnist.py worker <rank> <hostname> --algorithm <edp|syncps>
    
Examples:
    python train_mnist.py server mini1 --algorithm edp
    python train_mnist.py worker 1 mini2 --algorithm syncps
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

from smolcluster.algorithms.EDP.server import run_edp_server
from smolcluster.algorithms.EDP.worker import run_edp_worker
from smolcluster.algorithms.SynchronousPS.server import run_syncps_server
from smolcluster.algorithms.SynchronousPS.worker import run_syncps_worker
from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.device import get_device


def load_configs(algorithm: str = "syncps"):
    """Load configuration files.
    
    Args:
        algorithm: Either 'edp' or 'syncps' to determine which cluster config to load
    """
    CONFIG_DIR = Path(__file__).parent  / "configs"
    
    with open(CONFIG_DIR / "nn_config.yaml") as f:
        nn_config = yaml.safe_load(f)
    
    # Load appropriate cluster config based on algorithm
    config_file = f"cluster_config_{algorithm}.yaml"
    with open(CONFIG_DIR / config_file) as f:
        cluster_config = yaml.safe_load(f)
    
    return nn_config, cluster_config


def load_data(batch_size: int, world_size: int, seed: int, rank: int):
    """Load MNIST dataset for the given rank."""
    import torchvision
    from smolcluster.utils.data import get_data_indices
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])
    
    data_dir = Path(__file__).parent / "src" / "data"
    data = torchvision.datasets.MNIST(str(data_dir), download=True, transform=transforms)
    
    lendata = len(data)
    torch.manual_seed(seed)
    trainset, testset = torch.utils.data.random_split(
        data, [int(0.9 * lendata), lendata - int(0.9 * lendata)]
    )
    
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    
    batch_indices = get_data_indices(len(trainset), world_size, seed)
    train_data = torch.utils.data.Subset(trainset, batch_indices[rank])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


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
    """Run server for MNIST training.
    
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
    nn_config, cluster_config = load_configs(algorithm)
    
    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    rank = 0  # Server is rank 0
    batch_size = nn_config["batch_size"]
    
    # Create model
    model = SimpleMNISTModel(
        input_dim=nn_config["model"]["input_dim"],
        hidden=nn_config["model"]["hidden"],
        out=nn_config["model"]["out"],
    )
    device = get_device()
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, val_loader = load_data(batch_size, world_size, seed, rank)
    logger.info(f"Data ready. Train size: {len(train_loader)}, Val size: {len(val_loader)}")
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize W&B
    algo_name = algorithm.upper()
    wandb.init(
        project="smolcluster",
        name=f"{algo_name}-server-{hostname}_lr{nn_config['learning_rate']}_bs{batch_size}_workers{num_workers}",
        config={
            **nn_config,
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
            config=nn_config,
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
            config=nn_config,
            cluster_config=cluster_config,
            hostname=hostname,
            device=device,
            criterion=criterion,
        )
    wandb.finish()


def run_worker(worker_rank: int, hostname: str, algorithm: str = "syncps"):
    """Run worker for MNIST training.
    
    Args:
        worker_rank: Worker rank (1-indexed)
        hostname: Worker hostname
        algorithm: Either 'edp' or 'syncps'
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(f"[WORKER-{worker_rank}-MAIN]")
    
    setup_wandb()
    
    # Load configs
    nn_config, cluster_config = load_configs(algorithm)
    
    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    
    # Worker rank is 0-indexed internally but 1-indexed in command-line
    local_rank = worker_rank - 1
    
    # Get server connection info
    host_ip = cluster_config["host_ip"][hostname]
    port = cluster_config["port"]
    
    # Create model
    model = SimpleMNISTModel(
        input_dim=nn_config["model"]["input_dim"],
        hidden=nn_config["model"]["hidden"],
        out=nn_config["model"]["out"],
    )
    device = get_device()
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")
    
    # Print model summary
    logger.info("Model Summary:")
    summary = torchinfo.summary(model, input_size=(nn_config["batch_size"], 784), device=device)
    logger.info(f"\n{summary}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, val_loader = load_data(nn_config["batch_size"], world_size, seed, local_rank)
    logger.info(f"Data ready. Train size: {len(train_loader)}, Val size: {len(val_loader)}")
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize W&B
    algo_name = algorithm.upper()
    wandb.init(
        project="smolcluster",
        name=f"{algo_name}-worker-{hostname}_rank{local_rank}_lr{nn_config['learning_rate']}_bs{nn_config['batch_size']}",
        config={
            **nn_config,
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
            config=nn_config,
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
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=nn_config,
            cluster_config=cluster_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
        )
    wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed MNIST Training with EDP or SyncPS")
    parser.add_argument("mode", choices=["server", "worker"], help="Run as server or worker")
    parser.add_argument("hostname", help="Hostname of this node")
    parser.add_argument("rank", nargs="?", type=int, help="Worker rank (1-indexed, required for worker mode)")
    parser.add_argument("-a", "--algorithm", choices=["edp", "syncps"], default="syncps",
                        help="Training algorithm to use (default: syncps)")
    
    # Handle both new argparse format and legacy positional format
    if len(sys.argv) >= 2 and sys.argv[1] in ["server", "worker"]:
        # Try to parse with argparse first
        if "--algorithm" in sys.argv or "-a" in sys.argv:
            args = parser.parse_args()
            mode = args.mode
            hostname = args.hostname
            algorithm = args.algorithm
            worker_rank = args.rank
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
