import logging
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import yaml
import wandb

from transformers import AutoConfig, GPT2LMHeadModel

from smolcluster.utils.common_utils import (
    get_gradients,
    receive_message,
    send_message,
)
from smolcluster.utils.device import get_device
from smolcluster.utils.layers import (
    get_model_per_node
)
from smolcluster.utils.logging_utils import setup_cluster_logging


def evaluate(
    device: torch.device, 
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    criterion: torch.nn.Module,
    decoder_type_ppl: bool = False
) -> tuple[float, Optional[float]]:
    """Evaluate model on validation set."""
    model.eval()
    total_val_loss = 0.0
 
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            B, T, C = output.shape
            output = output.view(B*T, C)
            target = target.view(B*T)
            loss = criterion(output, target)
            total_val_loss += loss.item()
    
    avg_loss = total_val_loss / len(val_loader)
    ppl = math.exp(avg_loss) if decoder_type_ppl else None
    
    return avg_loss, ppl


def compute_worker_activations(
    device: torch.device,
    model: torch.nn.Module,
    data: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Compute activations for worker node."""
    model.train()
    data = data.to(device)
    hidden = model(data)
    return hidden


def compute_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    """Compute loss for given data and target."""
    model.eval()
    data, target = data.to(get_device()), target.to(get_device())
    output = model(data)
    B, T, C = output.shape
    output = output.view(B*T, C)
    target = target.view(B*T)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    grads = get_gradients(loss, model)
    return loss, grads

# Setup logging (will be replaced by setup_cluster_logging in run_modelparallelism_worker)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[WORKER]")


def connect_to_server(
    host: str, port: int, max_retries: int = 60, retry_delay: float = 3.0
) -> socket.socket:
    """Connect to server with retry logic."""
    # Ping to warm up ARP cache (especially important for WiFi networks)
    logger.info(f"Warming up ARP cache by pinging {host}...")
    try:
        subprocess.run(
            ["ping", "-c", "3", "-W", "1000", host], capture_output=True, timeout=10
        )
    except Exception as e:
        logger.warning(f"ARP warmup ping failed: {e}")

    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout for connection
        try:
            sock.connect((host, port))
            sock.settimeout(None)  # Remove timeout after connection
            logger.info(
                f"Connected to server at {host}:{port} on attempt {attempt + 1}"
            )
            return sock
        except (OSError, ConnectionRefusedError, socket.timeout) as e:
            sock.close()  # Close the failed socket
            # Re-ping every 5 attempts to keep ARP fresh
            if attempt > 0 and attempt % 5 == 0:
                logger.info(f"Re-pinging {host} to refresh ARP cache...")
                try:
                    subprocess.run(
                        ["ping", "-c", "2", "-W", "1000", host],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
            if attempt < max_retries - 1:
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to server after {max_retries} attempts"
                )
                raise
    # This should never be reached, but just in case
    raise RuntimeError("Failed to connect to server")


def run_modelparallelism_worker(
    model,
    optimizer,
    train_loader,
    val_loader,
    config,
    cluster_config,
    worker_rank,
    hostname,
    device,
    criterion,
    host_ip,
    port,
):
    """
    Run Model Parallelism worker for distributed GPT training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        worker_rank: Worker rank (1-indexed)
        hostname: Worker hostname
        device: Device to run on
        criterion: Loss criterion
        host_ip: Server IP address
        port: Server port
    """
    global logger
    
    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="worker",
        rank=worker_rank,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs")
    )
    logger.info(f"🚀 ModelParallelism Worker {worker_rank} starting up")
    
    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config.get("track_gradients", False)
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    
    # Set parameters
    local_rank = worker_rank + 1
    num_workers = cluster_config["num_workers"]
    num_nodes = cluster_config["num_nodes"]
    model_name = cluster_config["model_name"]
    
    # Use provided host_ip and port (from train.py)
    HOST_IP = host_ip
    PORT = port
    
    # Update logger with rank
    logger = logging.getLogger(f"[WORKER-{local_rank}]")
    
    logger.info(f"Worker {local_rank} starting. Connecting to server at {HOST_IP}:{PORT}")
    
    # Initialize model
    model = model.to(get_device())
    logger.info(f"Model initialized on device: {get_device()}")
    
    # Load model layers for this worker
    num_layers = config['num_layers']
    logger.info(f"Loading worker's share of model layers (rank {local_rank})...")
    
    model_layers, out_layers = get_model_per_node(
        model,
        num_nodes=num_nodes,
        local_rank=local_rank,
        total_layers=num_layers
    )
    
    model_layers = model_layers.to(get_device())
    logger.info(f"Loaded {len(model_layers)} layers for worker {local_rank}")
    
    # Connect to server
    sock = connect_to_server(HOST_IP, PORT)
    
    # Register with the server
    logger.info(f"Registering as worker {local_rank} with server...")
    send_message(sock, ("register", local_rank))

    while True:
        recv_command = receive_message(sock)

        if recv_command == "start_training":
            logger.info("Received start_training command from server.")
            break

    logger.info("Starting training loop...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            logger.info(f"[Step {step} / {num_epochs * len(train_loader)}] Waiting for activations from server")
            
            # Receive activations from server/previous worker
            message = receive_message(sock)
            command, recv_step, payload = message
            
            assert recv_step == step, f"Step mismatch: expected {step}, got {recv_step}"
            
            if command == 'generate_activations_train':
                logger.info(f"[Step {step}] Received command to generate activations for rank {local_rank}.")
                
                # Get activations from previous node
                out = payload['activations'].to(get_device())
                
                # Forward through this worker's layers
                for layer in model_layers: 
                    output = layer(out)
                    out = output[0] if isinstance(output, tuple) else output
        
                logger.info(f"[Step {step}] Finished generating activations for local_rank {local_rank}")
            
                logger.info(f"[Step {step}] Sending activations from rank {local_rank} to rank {local_rank + 1}")
                
                # Send activations to next worker/server
                send_message(sock, ('forward_activations', step, {
                    "from_rank": local_rank, 
                    "to_rank": local_rank + 1, 
                    "activations": out.cpu()
                }))
                
                del out
            
            elif command == 'generate_gradients':
                logger.info(f"[Step {step}] Received command to compute gradients for rank {local_rank}.")
                
                loss, grads =  compute_loss(model, data, target)
                
                logger.info(f"[Step {step}] Sending gradients from rank {local_rank} to rank {local_rank - 1}")
                send_message(sock, ('forward_gradients', step, {
                    "from_rank": local_rank, 
                    "to_rank": local_rank - 1, 
                    "gradients": grads.cpu()
                }))
            
            elif command == 'forward_gradients':
                
                rank, recv_grads = payload["to_rank"], payload["gradients"]
                logger.info(f"[Step {step}] Received gradients for rank {rank}.")  
                
                if rank == local_rank:
                    logger.info(f"[Step {step}] Computing backward pass for rank {local_rank}")
                    loss, grads = compute_loss(model, data, target)
                    loss.backward(recv_grads)
                    total_loss += loss.item()
                
            elif command == 'down':
                logger.info("Received exit command from server. Shutting down.")
                break
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Log gradient norms if tracking enabled
            if track_gradients:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log({
                            f"gradients/layer_{name}": grad_norm,
                            "step": step,
                            "epoch": epoch + 1,
                        })
            
            # Evaluation
            if step % eval_steps == 0:
                val_loss, val_ppl = evaluate(device, model, val_loader, criterion, decoder_type_ppl)
                
                if decoder_type_ppl:
                    wandb.log({
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                        "ppl/val": val_ppl,
                    })
                    logger.info(
                        f"[Step {step}] Evaluation: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}"
                    )
                else:
                    wandb.log({
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                    })
                    logger.info(
                        f"[Step {step}] Evaluation: Val Loss={val_loss:.4f}"
                    )
                model.train()
        
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed."
        )
        
    sock.close()
    logger.info("Worker training completed and connection closed.")



if __name__ == "__main__":
    main()
