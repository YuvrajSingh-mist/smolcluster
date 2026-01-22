import gc
import heapq
import logging
import math
import socket
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer


from smolcluster.utils.common_utils import (
    get_gradients,
    receive_message,
    send_message,
)
from smolcluster.utils.device import get_device
from smolcluster.utils.decoding import sample_next_token
from smolcluster.utils.model_downloader import ensure_model_weights
from smolcluster.utils.layers import (
    get_model_per_node
    
)
from smolcluster.utils.logging_utils import setup_cluster_logging


def compute_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for given data and target."""
    model.eval()
    data, target = data.to(get_device()), target.to(get_device())
    output = model(data)
    B, T, C = output.shape
    output = output.view(B*T, C)
    target = target.view(B*T)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    return loss

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


def compute_leader_activations(
    device: torch.device,
    model: torch.nn.Module,
    data: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute gradients for leader/server node."""
    optimizer.zero_grad()
    model.train()
    data = data.to(device)
    hidden = model(data)
    # B, T, C = output.shape
    # output = output.view(B*T, C)
    # target = target.view(B*T)
    # loss = criterion(output, target)
    # loss.backward()
    
    # Gradient clipping
    # if config.get("gradient_clipping", {}).get("enabled", False):
    #     max_norm = config["gradient_clipping"].get("max_norm", 1.0)
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # grads = get_gradients(model)
    return hidden


# Setup logging (will be replaced by setup_cluster_logging in run_modelparallelism_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[LEADER]")


def run_modelparallelism_server(
    model,
    optimizer,
    train_loader,
    val_loader,
    config,
    cluster_config,
    hostname,
    device,
    criterion,
):
    """
    Run Synchronous Parameter Server training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        hostname: Server hostname
        device: Device to run on
        criterion: Loss criterion
    """
    global logger
    
    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="server",
        rank=None,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs")
    )
    logger.info("🚀 ModelParallelism Server starting up")
    
    
    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config["track_gradients"]
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    num_workers = cluster_config["num_workers"]
    world_size = num_workers + 1
    model_name = cluster_config["model_name"]
     
    RANK = 0
    NUM_WORKERS = cluster_config['num_workers']    
    
    num_nodes = cluster_config['num_nodes']
    
     # Create socket
    HOST_IP = "0.0.0.0"
    PORT = cluster_config["port"]
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST_IP, PORT))
    sock.listen(5)
    # logger.info(f"Server listening on {HOST_IP}:{PORT}")
    
    logger.info(f"Server will bind to IP: {HOST_IP}, Port: {PORT}")
    workers = {}

    # Load tokenizer
    model = model.to(get_device())
    logger.info(f"Model initialized on device: {get_device()}")
    
    # Load model layers for server (rank 0)
    num_layers = cluster_config['num_layers']
    logger.info(f"Loading server's share of model layers (rank {RANK})...")
    
    model_layers, out_layers = get_model_per_node(
        model,
        num_nodes=num_nodes,
        local_rank=RANK,
        total_layers=num_layers
    )
    
   
    
    model_layers = model_layers.to(get_device())
    logger.info(f"Server loaded {len(model_layers)} layers")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the host and port
    sock.bind((HOST_IP, PORT))
    
    # Listen for incoming connections
    sock.listen(5)
    logger.info(f"Server listening on {HOST_IP}:{PORT}")
   
    logger.info(f"Server is running at {HOST_IP}:{PORT}")
    
    # Accept connections and wait for registration
    # Use priority queue to maintain workers sorted by rank
    worker_queue = []  # Priority queue: [(rank, socket, address)]
    registered_workers = {}  # rank -> socket (for quick lookup)
    client_socket = None  # API client socket
    
    # Accept all connections (workers + API client)
    while len(registered_workers) < NUM_WORKERS or client_socket is None:
        conn, address = sock.accept()
        logger.info(f"Accepted connection from {address}")

        # Wait for registration message
        try:
            message = receive_message(conn)
            if message is None:
                logger.warning(
                    f"Connection from {address} closed before registration"
                )
                conn.close()
                continue

            command, rank = message
            if command == "register":
                logger.info(f"Worker rank {rank} registered from {address}")
                registered_workers[rank] = conn
                workers[address] = conn
                # Add to priority queue sorted by rank
                heapq.heappush(worker_queue, (rank, conn, address))
                logger.info(f"Worker rank {rank} added to priority queue (queue size: {len(worker_queue)})")
            
            else:
                logger.warning(f"Unexpected message from {address}: {command}")
                conn.close()
        except Exception as e:
            logger.error(f"Error during registration from {address}: {e}")
            conn.close()
            continue

    logger.info(f"All workers connected. Starting inference on {model_name}...")
    logger.info(f"Worker priority queue (by rank): {[(rank, addr) for rank, _, addr in worker_queue]}")

    # Send start_inference to workers in rank order
    for rank, worker_socket, addr in sorted(worker_queue):
        logger.info(f"Sending start_inference to worker rank {rank} at {addr}")
        send_message(worker_socket, "start_training")

    logger.info(f"Starting inference for {model_name}.")
    logger.info("Waiting for inference requests from API client...")
    
    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            logger.info(f"[Step {step}  / {num_epochs * len(train_loader)}] Server computing leader activations")
            leader_activations = compute_leader_activations(
                device, model, data, target, config
            )
            total_loss += leader_activations.item()
            logger.info(f"[Step {step}] Leader activations: {leader_activations.item():.4f}")
       
            logger.info(f"Generating activations for input IDs for rank 0")

            with torch.no_grad():
    
                out = model_layers[0](leader_activations)
            
                pos_ids = torch.arange(out.shape[1], dtype=torch.long, device=get_device())
                out = out + model_layers[1](pos_ids)
                
                for layer in model_layers[2:]:
                    output = layer(out)
                    out = output[0] if isinstance(output, tuple) else output
            
            logger.info("Finsihed generating activations for local_rank 0")
            
            activations = out.cpu()
            # Send generation request to all workers in rank order (1, 2, ...)
            for rank, worker_socket, addr in sorted(worker_queue):
                
                send_message(
                    worker_socket,
                    (
                        "generate_activations_train",
                        {
                          
                            "activations": activations,

                        },
                    ),
                )

                message = receive_message(worker_socket)
                
                command, payload = message
                
                if command == 'forward_activations':
                    activations = payload['activations'].to(get_device())
                    from_rank = payload['from_rank']
                    to_rank = payload['to_rank']
                    logger.info(f"Received activations forwarded from worker {from_rank} to worker {to_rank}")
                    
                else:
                    logger.error(f"Unexpected command from worker {rank}: {command}. Cannot continue.")
                    break
            
            for rank, worker_socket, addr in sorted(worker_queue, reverse=True):
                
                
                if rank == NUM_WORKERS - 1:
                    send_message(
                        worker_socket,
                        (
                            "generate_gradients"
                        ),
                    )
                else:
                    send_message(
                        worker_socket,
                        (
                            "forward_gradients",
                            {
                                "gradients": activations,
                            },
                        ),
                    )
                #Receiving the last worker nodes activations
                message = receive_message(sock)

                command, payload = message
                if command == 'forward_gradients':
                    grads = payload['gradients'].to(get_device())
                    logger.info(f"Received gradients from last worker node.")
                    if rank == RANK:
                        loss = compute_loss(model, data, target)
                        loss.backward(grads)
                    
                    send_message(worker_socket, ("forward_gradients", grads))
            
            optimizer.step()    
            
            del activations 
            
            gc.collect()
            activations = None
    
    for rank, worker_socket, addr in sorted(worker_queue):
        
        send_message(worker_socket, "down")
        
    logger.info("Training completed successfully!")
       
    sock.close()

