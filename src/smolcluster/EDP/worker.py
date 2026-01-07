import logging
import socket
import subprocess
import sys
import time

import torch
import torchvision
import yaml

from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.common_utils import (
    get_gradients,
    receive_message,
    send_message,
    set_gradients, set_weights)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device

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
update_freq = cluster_config["low_worker_update_freq"]

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
local_rank = int(WORKER_RANK)

# Workers connect to the server using the IP specified for this worker's hostname
HOST_IP = cluster_config["host_ip"][HOSTNAME]
batch_size = nn_config["batch_size"]
num_epochs = nn_config["num_epochs"]
eval_steps = nn_config["eval_steps"]
gradient_scaling = nn_config.get("gradient_scaling", 0.0)

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
    
    optimizer = torch.optim.SGD(model.parameters(), lr=nn_config["learning_rate"])

    while True:
        recv_command = receive_message(sock)

        if recv_command == "start_training":
            logger.info("Received start_training command from server.")
            break

    for step in range(total_steps):
        model.train()

        total_loss = 0.0
        epoch = step // len(train_loader)
        batch_idx = step % len(train_loader)
        data = train_loader.dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        data, target = zip(*data)
        
        logger.info("Performing local forward and backward pass.")
        # for batch_idx, (data, target) in enumerate(train_loader):
            # step = epoch * len(train_loader) + batch_idx
        optimizer.zero_grad()
        data, target = data.to(get_device()), target.to(get_device())

        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)

        loss.backward()
        grads = get_gradients(model)

        # Send gradients to server with version
        send_message(sock, (
            "parameter_server_reduce",
            {
                "step": step,
                "rank": local_rank,
                "grads": grads,
                "model_version": model_version,
            }
        ))

        # Wait for signal to pull weights
        data_recv = receive_message(sock)

        command, new_version = data_recv

        if command == "pull_weights":
            logger.info(f"Pulling weights (version {new_version})")
            send_message(sock, ("pull_weights", model_version))
            weights, new_version = receive_message(sock)
            set_weights(weights, model)
            model_version = new_version
            logger.info(f"Updated to model version {model_version}")

        total_loss += loss.item()
        
        logger.info(
            f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)} completed."
        )
        

if __name__ == "__main__":
    main()
