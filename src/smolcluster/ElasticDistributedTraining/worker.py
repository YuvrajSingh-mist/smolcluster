import logging
import os
import socket
import subprocess
import sys
import time

import torch
import torchvision
import wandb
import yaml

from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.common_utils import (
    get_gradients,
    receive_message,
    send_message,
    set_weights,
    get_weights)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device
from smolcluster.utils.quantization import dequantize_model_weights, quantize_model_weights, calculate_compression_ratio

# Login to wandb using API key from environment variable
if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    logger_temp = logging.getLogger("[WORKER-INIT]")
    logger_temp.info("✅ Logged into wandb using WANDB_API_KEY")
else:
    logger_temp = logging.getLogger("[WORKER-INIT]")
    logger_temp.warning("⚠️  WANDB_API_KEY not set - wandb may prompt for login")

# Load configs
with open("../configs/nn_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open("../configs/cluster_config_edp.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
PORT = cluster_config["port"]
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1
worker_update_interval = cluster_config.get("worker_update_interval", 5)

# Get worker rank and hostname from command-line arguments
if len(sys.argv) > 1:
    WORKER_RANK = sys.argv[1]
else:
    WORKER_RANK = input(f"Enter worker ID (1 to {NUM_WORKERS}): ")

if len(sys.argv) > 2:
    HOSTNAME = sys.argv[2]
else:
    HOSTNAME = input("Enter worker hostname: ")

# Set parameters
local_rank = int(WORKER_RANK) - 1

# Workers connect to the server using the IP specified for this worker's hostname
HOST_IP = cluster_config["host_ip"][HOSTNAME]
batch_size = nn_config["batch_size"]
num_epochs = nn_config["num_epochs"]
eval_steps = nn_config["eval_steps"]
recv_model_version = -1
# Loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Track model version for elastic training
model_version = 0

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(f"[WORKER-{local_rank}]")

logger.info(f"Worker {local_rank} starting. Connecting to server at {HOST_IP}:{PORT}")


def load_data(batch_size, WORLD_SIZE, SEED, local_rank):
    # load MNIST
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data = torchvision.datasets.MNIST("../../data", download=True, transform=transforms)
    lendata = len(data)
    torch.manual_seed(SEED)
    trainset, testset = torch.utils.data.random_split(
        data, [int(0.9 * lendata), lendata - int(0.9 * lendata)]
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    batch_indices = get_data_indices(len(trainset), WORLD_SIZE, SEED)
    train_data = torch.utils.data.Subset(trainset, batch_indices[local_rank])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(get_device()), target.to(get_device())
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


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


def main():
    
    global model_version, recv_model_version
    
    # Initialize W&B for worker
    wandb.init(
        project="smolcluster",
        name=f"worker-{local_rank}_lr_{nn_config['learning_rate']}_bsz_{nn_config['batch_size']}",
        config={
            **nn_config,
            "worker_rank": local_rank,
            "worker_update_interval": cluster_config.get("worker_update_interval", 5),
        },
    )
    logger.info(f"Worker {local_rank} wandb initialized")
    
    # Connect to server with retry logic
    sock = connect_to_server(HOST_IP, PORT)

    # Register with the server
    logger.info(f"Registering as worker {local_rank} with server...")
    send_message(sock, ("register", local_rank))

    model = SimpleMNISTModel(
        input_dim=nn_config["model"]["input_dim"],
        hidden=nn_config["model"]["hidden"],
        out=nn_config["model"]["out"],
    )
    model = model.to(get_device())
    logger.info(f"Model initialized on device: {get_device()}")

    train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, local_rank)
    logger.info(
        f"Data loaders ready. Train size: {len(train_loader)}, Test size: {len(val_loader)}"
    )
    
    total_steps = num_epochs * len(train_loader)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])

    # Wait for start signal or timeout and start anyway
    logger.info("Waiting for start_training signal from server (max 5 seconds)...")
    start_time = time.time()
    while time.time() - start_time < 5:
        sock.settimeout(0.1)
        try:
            recv_command = receive_message(sock)
            if recv_command == "start_training":
                logger.info("Received start_training command from server.")
                break
        except socket.timeout:
            pass
    
    logger.info("Starting training loop...")
    sock.settimeout(None)  # Reset to blocking

    # Initialize iterator for continuous training
    train_iter = iter(train_loader)
 
    for step in range(total_steps):
        model.train()
        epoch = step // len(train_loader)
        
        # Fetch next batch, cycling through dataset
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        
        logger.info("Performing local forward and backward pass.")
        optimizer.zero_grad()
        data, target = data.to(get_device()), target.to(get_device())

        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        
        # NEW APPROACH: Send quantized model weights for Polyak averaging (75% compression)
        weights = get_weights(model)
        quantized_weights = quantize_model_weights(weights)
        
        # Log compression ratio on first step
        if step == 0:
            comp_info = calculate_compression_ratio(weights, quantized_weights)
            logger.info(f"Quantization: {comp_info['original_mb']:.2f}MB → {comp_info['compressed_mb']:.2f}MB (ratio: {comp_info['ratio']:.2f}x)")
        
        logger.info("Local forward and backward pass done. Sending quantized model weights to server.")
        send_message(sock, (
            "parameter_server_reduce",
            {
                "step": step,
                "rank": local_rank,
                "quantized_weights": quantized_weights,  # Send quantized weights
                "model_version": model_version,
            }
        ))
        logger.info("Quantized model weights sent to server.")
        
        # OLD APPROACH: Send gradients for scaling (commented out)
        # grads = get_gradients(model)
        # send_message(sock, (
        #     "parameter_server_reduce",
        #     {
        #         "step": step,
        #         "rank": local_rank,
        #         "grads": grads,
        #         "model_version": model_version,
        #     }
        # ))
        
        if step % worker_update_interval == 0 and step != 0:
            logger.info(f"Pulling weights from server at step {step}.")
            send_message(sock, ("pull_weights", model_version))
            sock.settimeout(1.0)  # Wait up to 1 second for weights
            try:
                weights, new_version = receive_message(sock)
                
                dequant_weights = dequantize_model_weights(weights, device=get_device())
                model.load_state_dict(dequant_weights)
                
                recv_model_version = new_version
                logger.info(f"Updated to model version {recv_model_version} from server.")
            except socket.timeout:
                logger.warning("Timeout while pulling weights from server.")
            except BlockingIOError:
                logger.error(f"non-blocking socket error while pulling weights from server.")
            finally:
                sock.settimeout(None)  # Restore blocking socket

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
                
        # Update local model version if received new weights
        if recv_model_version != -1 and recv_model_version != model_version:
            model_version = recv_model_version
            logger.info(f"Updated local model version to {model_version}.")
        
        logger.info(
            f"Epoch: {epoch} , Step {step}/{total_steps} completed."
        )
    
    send_message(sock, ("disconnect", local_rank))
    sock.close()
    logger.info(f"Training complete. Worker {local_rank} disconnected.")
    
    # Finish wandb tracking
    wandb.finish()
    logger.info(f"Worker {local_rank} wandb tracking finished.")

if __name__ == "__main__":
    main()
