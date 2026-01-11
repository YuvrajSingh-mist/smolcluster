import gc
import logging
import os
import socket
import sys
import threading
import time
from typing import Tuple

import torch
import torchinfo
import torchvision
import wandb
import yaml
from torch.utils.data import DataLoader

from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.common_utils import (
    get_gradients,
    receive_message,
    send_message,
    set_gradients,
    get_weights,
    set_weights,
    
)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device
from smolcluster.utils.quantization import dequantize_model_weights, quantize_model_weights

# Login to wandb using API key from environment variable
if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    logger_temp = logging.getLogger("[SERVER-INIT]")
    logger_temp.info("✅ Logged into wandb using WANDB_API_KEY")
else:
    logger_temp = logging.getLogger("[SERVER-INIT]")
    logger_temp.warning("⚠️  WANDB_API_KEY not set - wandb may prompt for login")

# Get hostname from command-line argument
if len(sys.argv) > 1:
    HOSTNAME = sys.argv[1]
else:
    HOSTNAME = input("Enter server hostname: ")

# Load configs
with open("../configs/nn_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open("../configs/cluster_config_edp.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
HOST_IP = "0.0.0.0"  # Listen on all network interfaces (Thunderbolt, eth0, WiFi, etc.)
PORT = cluster_config["port"]
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1
RANK = 0

batch_size = nn_config["batch_size"]
eval_steps = nn_config["eval_steps"]
num_epochs = nn_config["num_epochs"]
track_gradients = nn_config.get("track_gradients", False)
use_quantization = cluster_config.get("use_quantization", True)

criterion = torch.nn.CrossEntropyLoss()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SERVER]")
logger.info(f"Server will bind to IP: {HOST_IP}, Port: {PORT}")

gradients_event = threading.Event()
lock = threading.Lock()

model_version = 0  # Track global model version for elastic training

workers = {}
workers_grads_received = {}  # Single dict for all worker gradients: {(rank, recv_step, worker_version): grads}


def load_data(
    batch_size: int, WORLD_SIZE: int, SEED: int, RANK: int
) -> tuple[DataLoader, DataLoader]:
    # load MNIST
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data = torchvision.datasets.MNIST("../data", download=True, transform=transforms)
    lendata = len(data)
    torch.manual_seed(SEED)
    trainset, testset = torch.utils.data.random_split(
        data, [int(0.9 * lendata), lendata - int(0.9 * lendata)]
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    batch_indices = get_data_indices(len(trainset), WORLD_SIZE, SEED)
    train_data = torch.utils.data.Subset(trainset, batch_indices[RANK])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def evaluate(
    model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    val_iter = iter(val_loader)
    
    with torch.no_grad():
        for step in range(len(val_loader)):
            
            try:
                data, target = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                data, target = next(val_iter)
            
            data, target = data.to(get_device()), target.to(get_device())
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * (correct / total)
    model.train()
    return avg_loss, accuracy


def compute_leader_gradients(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    model.train()
    data, target = data.to(get_device()), target.to(get_device())
    output = model(data.view(data.size(0), -1))
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    grads = get_gradients(model)
    return grads


def handle_worker(conn: socket.SocketType, addr: tuple[str, int]) -> None:
    global model_version
    logger.info(f"Handling worker at {addr}")

    while True:
        
        message = receive_message(conn)

        # Handle connection closed or empty message
        if message is None:
            logger.info(f"Worker {addr} closed connection")
            break

        # Unpack the message tuple
        command, payload = message

        if command == "polyark_averaging":
            recv_step = payload["step"]
            rank = payload["rank"]
            worker_version = payload["model_version"]
            
            # Check if worker sent quantized weights, weights, or gradients
            if "quantized_weights" in payload:
                # New approach: Dequantize and use Polyak averaging
                quantized_weights = payload["quantized_weights"]
                logger.info(
                    f"Received quantized weights from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                )
                # Dequantize weights back to float32 on the server's device
                device_str = str(get_device())
                weights = dequantize_model_weights(quantized_weights, device=device_str)
                with lock:
                    workers_grads_received[(rank, recv_step, worker_version)] = {"type": "weights", "data": weights}
            elif "weights" in payload:
                # Legacy: Full float32 weights (no compression)
                weights = payload["weights"]
                logger.info(
                    f"Received model weights from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                )
                with lock:
                    workers_grads_received[(rank, recv_step, worker_version)] = {"type": "weights", "data": weights}
            else:
                # Old approach: Gradient scaling (kept for reference)
                grads = payload["grads"]
                logger.info(
                    f"Received gradients from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                )
                with lock:
                    workers_grads_received[(rank, recv_step, worker_version)] = {"type": "grads", "data": grads}
                
            gradients_event.set()
            logger.info(f"Data stored successfully for worker {rank} at step {recv_step}")
        
        elif command == 'pull_weights':
            worker_version = payload  # payload is the worker's current model version
            logger.info(f"Worker {addr} requested weights (worker version: {worker_version}, server version: {model_version})")
            
            weights = get_weights(model)
            if use_quantization:
                quantized_weights = quantize_model_weights(weights)
                send_message(conn, (quantized_weights, model_version))
                logger.info(f"Quantized weights sent to worker {addr}")
            else:
                send_message(conn, (weights, model_version))
                logger.info(f"Weights sent to worker {addr}")
            
        elif command == 'disconnect':
            logger.info(f"Worker {addr} requested disconnection.")
            #  Remove disconnected worker
            with lock:
                workers.pop(addr, None)
            
        
    conn.close()
    

def polyak_average_weights(
    current_weights: dict[str, torch.Tensor],
    worker_weights: dict[str, torch.Tensor],
    staleness: int,
    # alpha_base: float = 1.0
) -> Tuple[dict[str, torch.Tensor], float]:
    """
    Blend current model with worker's model using staleness-aware Polyak averaging.
    
    Args:
        current_weights: Current server model weights
        worker_weights: Worker's trained model weights
        staleness: |current_version - worker_version|
        alpha_base: Base weight for worker model (default: 1.0)
    
    Returns:
        Blended model weights
    """
    # Worker weight decreases with staleness
    staleness_factor = 1 / (1 + staleness)
    
    blended = {}
    for name in current_weights.keys():
        blended[name] = (
            staleness_factor * worker_weights[name] + 
            (1.0 - staleness_factor) * current_weights[name]
        )
    
    return blended, staleness_factor



model = SimpleMNISTModel(
    input_dim=nn_config["model"]["input_dim"],
    hidden=nn_config["model"]["hidden"],
    out=nn_config["model"]["out"],
)
model = model.to(get_device())
logger.info(f"Model initialized on device: {get_device()}")


train_loader, val_loader = load_data(batch_size, NUM_WORKERS, SEED, RANK)


logger.info(
    f"Data ready. Train size: {len(train_loader)}, Test size: {len(val_loader)}"
)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
sock.bind((HOST_IP, PORT))

# Listen for incoming connections
sock.listen(5)
logger.info(f"Server listening on {HOST_IP}:{PORT}")



def main():
    global model_version
    shutdown_flag = threading.Event()
    
    # Initialize W&B
    wandb.init(
        project="smolcluster",
        name=f"server-{HOSTNAME}_lr{nn_config['learning_rate']}_bs{nn_config['batch_size']}_workers{len(cluster_config['workers'])}",
        config={
            **nn_config,
            "server_hostname": HOSTNAME,
            "worker_hostnames": cluster_config['workers'],
            "num_workers": len(cluster_config['workers']),
        },
    )

    model_summary = str(
        torchinfo.summary(model, input_size=(batch_size, 784), device=get_device())
    )
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Start accepting worker connections in background
    def accept_workers():
        while not shutdown_flag.is_set():
            try:
                client_socket, client_address = sock.accept()
                logger.info(f"Accepted connection from {client_address}")
                
                # Wait for registration message
                message = receive_message(client_socket)
                if message is None:
                    logger.warning(f"Connection from {client_address} closed before registration")
                    client_socket.close()
                    continue

                command, rank = message
                if command == "register":
                    logger.info(f"Worker {rank} registered from {client_address}")
                    with lock:
                        workers[rank] = client_socket
                    threading.Thread(target=handle_worker, args=(client_socket, client_address), daemon=True).start()
                else:
                    logger.warning(f"Unexpected message from {client_address}: {command}")
                    client_socket.close()
            except OSError:
                # Socket closed, exit gracefully
                if shutdown_flag.is_set():
                    logger.info("Worker acceptance thread shutting down")
                    shutdown_flag.set()
                    break
                else:
                    logger.error("Socket error occurred")
            except Exception as e:
                if not shutdown_flag.is_set():
                    logger.error(f"Error accepting worker: {e}")
    
    # Start worker acceptance thread
    threading.Thread(target=accept_workers, daemon=True).start()
    logger.info("Worker acceptance thread started")
    
    # Give workers a moment to connect
    time.sleep(2)

    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])

    logger.info(f"Starting training for {num_epochs} epochs.")
    
    # Send start signal to all connected workers
    with lock:
        for worker_socket in workers.values():
            try:
                send_message(worker_socket, "start_training")
            except:
                pass
    
    train_iter = iter(train_loader)
    total_steps = num_epochs * len(train_loader)
    total_loss = 0.0
   
    for step in range(total_steps):
        model.train()
        
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        
        data = data.to(get_device())
        target = target.to(get_device())
        
        epoch = step // len(train_loader)
        
        # Compute leader gradients
        leader_grads = compute_leader_gradients(model, data, target, criterion, optimizer)
        logger.info(f"Epoch {epoch + 1}, Step: {step}: Computed leader gradients.")
        
        # Collect any available worker data (non-blocking with small timeout)
        gradients_event.wait(timeout=0.01)  # Very short wait
        gradients_event.clear()
        
        with lock:
            workers_copy = dict(workers_grads_received)
            workers_grads_received.clear()
        
        if workers_copy:
            logger.info(f"Step {step}: Collected {len(workers_copy)} worker update(s)")
            
            # Polyak averaging with model weights
            current_weights = get_weights(model)
            
            for (rank, recv_step, worker_version), worker_data in workers_copy.items():
                staleness = abs(model_version - worker_version)
                
                if worker_data["type"] == "weights":
                    # Polyak averaging: blend worker model with current model
                    worker_weights = worker_data["data"]
                    worker_weights = {k: v.to(get_device()) for k, v in worker_weights.items()}
                    current_weights = {k: v.to(get_device()) for k, v in current_weights.items()}
                    
                    blended_weights, staleness_factor = polyak_average_weights(
                        current_weights, worker_weights, staleness
                    )
                    
                    logger.info(
                        f"Applying worker {rank} model via Polyak averaging "
                        f"(staleness: {staleness}, alpha: {staleness_factor:.3f})"
                    )
                    
                    # Update model with blended weights
                    model.load_state_dict(blended_weights)
                    
                    current_weights = blended_weights  # Update for next worker
                    
                    with lock:
                        model_version += 1
                        
                else:
                    # OLD APPROACH: Gradient scaling (kept for backward compatibility)
                    # This code path is commented out but functional if workers send gradients
                    grads = worker_data["data"]
                    scale = 1.0 / (1e-8 + staleness)
                    
                    logger.info(
                        f"[DEPRECATED] Applying worker {rank} grads via scaling "
                        f"(staleness: {staleness}, scale: {scale:.3f})"
                    )
                    
                    optimizer.zero_grad()
                    scaled_grads = {k: v * scale for k, v in grads.items()}
                    set_gradients(scaled_grads, model)
                    optimizer.step()
                    
                    with lock:
                        model_version += 1
                
            del workers_copy
            gc.collect()
        
        # Apply leader gradients
        optimizer.zero_grad()
        set_gradients(leader_grads, model)
        optimizer.step()
        
        with lock:
            model_version += 1
        
        del leader_grads
        gc.collect()
        
        logger.info(f"Applied leader gradients. Step {step}: Updated to model version {model_version}")
        
        
        data = data.to(get_device())
        target = target.to(get_device())
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        total_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}, Step: {step}: Step loss = {loss.item():.4f}")
        
        if step % 50 == 0:
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "losses/step_loss": loss.item(),
                }
            )
            
            # if track_gradients:
            logger.info("Tracking gradients in wandb...")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.info(f"Logging gradients for layer: {name}")
                    grad_norm = torch.norm(param.grad.detach(), 2).item()
                    wandb.log(
                        {
                            f"gradients/layer_{name}": grad_norm,
                            "step": step,
                        }
                    )
            logger.info("Gradient tracking complete.")
            
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "lr": nn_config["learning_rate"],
                    "batch_size": nn_config["batch_size"],
                }
            )

        if step % eval_steps == 0:
            
            logger.info(f"Evaluating model at step {step}...")
            
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            wandb.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "losses/val": val_loss,
                    "accuracy/val": val_acc,
                }
            )

    logger.info(f"Training completed. Total steps: {step + 1}, Final model version: {model_version}")
    logger.info("Waiting for any remaining worker updates...")
    
    gradients_event.wait(timeout=0.01) 
    gradients_event.clear()
    
    while len(workers) > 0:
        
        gradients_event.wait(timeout=0.01)  
        gradients_event.clear()
    
        with lock:
            workers_copy = dict(workers_grads_received)
            workers_grads_received.clear()
        
        
        if workers_copy:
            logger.info(f"Step {step}: Collected {len(workers_copy)} worker update(s)")
            
            # NEW APPROACH: Polyak averaging with model weights
            current_weights = get_weights(model)
            
            for (rank, recv_step, worker_version), worker_data in workers_copy.items():
                staleness = abs(model_version - worker_version)
                
                if worker_data["type"] == "weights":
                    # Polyak averaging: blend worker model with current model
                    worker_weights = worker_data["data"]

                    worker_weights = {k: v.to(get_device()) for k, v in worker_weights.items()}
                    current_weights = {k: v.to(get_device()) for k, v in current_weights.items()}
                    
                    
                    blended_weights, staleness_factor = polyak_average_weights(
                        current_weights, worker_weights, staleness
                    )
                    
                    logger.info(
                        f"Applying worker {rank} model via Polyak averaging "
                        f"(staleness: {staleness}, alpha: {staleness_factor:.3f})"
                    )
                    
                    # Update model with blended weights
                    model.load_state_dict(blended_weights)
                    current_weights = blended_weights  # Update for next worker
                    
                    with lock:
                        model_version += 1
                        
                else:
                    # OLD APPROACH: Gradient scaling (kept for backward compatibility)
                    # This code path is commented out but functional if workers send gradients
                    grads = worker_data["data"]
                    scale = 1.0 / (1e-8 + staleness)
                    
                    logger.info(
                        f"[DEPRECATED] Applying worker {rank} grads via scaling "
                        f"(staleness: {staleness}, scale: {scale:.3f})"
                    )
                    
                    optimizer.zero_grad()
                    scaled_grads = {k: v * scale for k, v in grads.items()}
                    set_gradients(scaled_grads, model)
                    optimizer.step()
                    
                    with lock:
                        model_version += 1
                
            del workers_copy
            gc.collect()
        
            # # Apply worker's gradients
            # optimizer.zero_grad()
            # optimizer.step()
            
            with lock:
                model_version += 1
            
            step += 1
            
            gc.collect()
            
            data = data.to(get_device())
            target = target.to(get_device())
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item()
            
            logger.info(f"Step: {step}: Step loss = {loss.item():.4f}")
            
            if step % 50 == 0:
                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch,
                        "losses/step_loss": loss.item(),
                    }
                )
                
                
                
                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch,
                        "lr": nn_config["learning_rate"],
                        "batch_size": nn_config["batch_size"],
                    }
                )

            if step % eval_steps == 0:
                
                logger.info(f"Evaluating model at step {step}...")
                
                val_loss, val_acc = evaluate(model, val_loader, criterion)

                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch,
                        "losses/val": val_loss,
                        "accuracy/val": val_acc,
                    }
                )
        
        
        
    shutdown_flag.set()
    sock.close()
    logger.info("Server shutdown complete")
    
    wandb.finish()


if __name__ == "__main__":
    
    main()
