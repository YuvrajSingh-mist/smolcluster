#!/usr/bin/env python3
"""
MNIST Training with EDP (Elastic Distributed Parameter Server)

This script provides both server and worker entry points for distributed
MNIST training using the refactored EDP functions.

Usage:
    Server: python train_mnist.py server <hostname>
    Worker: python train_mnist.py worker <rank> <hostname>
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torchinfo
import wandb
import yaml
from dotenv import load_dotenv

load_dotenv()

from smolcluster.algorithms.EDP.server import run_edp_server
from smolcluster.algorithms.EDP.worker import run_edp_worker
from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.device import get_device


def load_configs():
    """Load configuration files."""
    CONFIG_DIR = Path(__file__).parent  / "configs"
    
    with open(CONFIG_DIR / "nn_config.yaml") as f:
        nn_config = yaml.safe_load(f)
    
    with open(CONFIG_DIR / "cluster_config_edp.yaml") as f:
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
    if "WANDB_API_TOKEN" in os.environ:
        wandb.login(key=os.environ["WANDB_API_TOKEN"], relogin=True)
        logger = logging.getLogger("[INIT]")
        logger.info("✅ Logged into wandb using WANDB_API_TOKEN")
    else:
        logger = logging.getLogger("[INIT]")
        logger.warning("⚠️  WANDB_API_TOKEN not set - wandb may prompt for login")


def run_server(hostname: str):
    """Run EDP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("[SERVER-MAIN]")
    
    setup_wandb()
    
    # Load configs
    nn_config, cluster_config = load_configs()
    
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
    
    # Run server
    logger.info("Starting EDP server...")
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


def run_worker(worker_rank: int, hostname: str):
    """Run EDP worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(f"[WORKER-{worker_rank}-MAIN]")
    
    setup_wandb()
    
    # Load configs
    nn_config, cluster_config = load_configs()
    
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
    
    # Run worker
    logger.info(f"Starting EDP worker {local_rank}...")
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


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python train_mnist.py server <hostname>")
        print("  Worker: python train_mnist.py worker <rank> <hostname>")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "server":
        if len(sys.argv) < 3:
            print("Error: Server mode requires hostname argument")
            print("Usage: python train_mnist.py server <hostname>")
            sys.exit(1)
        hostname = sys.argv[2]
        run_server(hostname)
    
    elif mode == "worker":
        if len(sys.argv) < 4:
            print("Error: Worker mode requires rank and hostname arguments")
            print("Usage: python train_mnist.py worker <rank> <hostname>")
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
