import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchvision
import wandb
import yaml

from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.common_utils import (
    get_gradients,
    get_weights,
    receive_message,
    send_message,
    set_weights,
)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device

# Login to wandb using API key from environment variable
if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    logger_temp = logging.getLogger("[WORKER-INIT]")
    logger_temp.info("✅ Logged into wandb using WANDB_API_KEY")
else:
    logger_temp = logging.getLogger("[WORKER-INIT]")
    logger_temp.warning("⚠️  WANDB_API_KEY not set - wandb may prompt for login")

# Load configs
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
with open(CONFIG_DIR / "nn_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open(CONFIG_DIR / "cluster_config_syncps.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
PORT = cluster_config["port"]
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1

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
worker_update_interval = cluster_config["worker_update_interval"]
polyak_alpha = nn_config["polyak_alpha"]
track_gradients = nn_config["track_gradients"]
# Loss criterion
criterion = torch.nn.CrossEntropyLoss()

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
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    data = torchvision.datasets.MNIST(
        str(DATA_DIR), download=True, transform=transforms
    )
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


def polyak_blend_weights(
    current_weights: dict[str, torch.Tensor],
    server_weights: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Blend current worker model with server's model using Polyak averaging.

    Args:
        current_weights: Current worker model weights
        server_weights: Server's averaged model weights
        alpha: Blending factor (0.5 = equal blend, higher = more server influence)

    Returns:
        Blended model weights
    """
    blended = {}
    for name in current_weights.keys():
        blended[name] = (
            alpha * server_weights[name] + (1.0 - alpha) * current_weights[name]
        )

    return blended


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
    # Initialize W&B for worker
    wandb.init(
        project="smolcluster",
        name=f"SyncPS-worker-{HOSTNAME}_rank{local_rank}_lr{nn_config['learning_rate']}_bs{nn_config['batch_size']}",
        config={
            **nn_config,
            "worker_rank": local_rank,
            "worker_hostname": HOSTNAME,
            "mode": "synchronous_ps",
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

    # optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])

    while True:
        recv_command = receive_message(sock)

        if recv_command == "start_training":
            logger.info("Received start_training command from server.")
            break

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            logger.info(f"[Step {step}] Starting forward and backward pass")
            # optimizer.zero_grad()
            data, target = data.to(get_device()), target.to(get_device())

            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()

            # Gradient clipping
            if nn_config.get("gradient_clipping", {}).get("enabled", False):
                max_norm = nn_config["gradient_clipping"].get("max_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                logger.info(
                    f"[Step {step}] Applied gradient clipping with max_norm={max_norm}"
                )

            grads = get_gradients(model)
            logger.info(f"[Step {step}] Computed local gradients")

            # Send gradients to server and receive updated weights
            logger.info(f"[Step {step}] Sending gradients to server")
            send_message(sock, ("parameter_server_reduce", step, local_rank, grads))

            logger.info(f"[Step {step}] Waiting for ACK and step from server")
            data_recv = receive_message(sock)

            command, recv_step, weights = data_recv
            logger.info(
                f"[Step {step}] Received '{command}' from server for step {recv_step}"
            )

            assert recv_step == step, "Step mismatch in communication with server."

            if command == "model_weights":
                # Apply Polyak averaging to blend server weights with local model
                current_weights = get_weights(model)
                alpha = polyak_alpha  # Use the loaded polyak_alpha value
                blended_weights = polyak_blend_weights(current_weights, weights, alpha)
                set_weights(blended_weights, model)
                logger.info(
                    f"[Step {step}] ✅ Applied Polyak-averaged weights (alpha={alpha})"
                )
            else:
                logger.warning(
                    f"[Step {step}] Expected 'model_weights' but got '{command}'"
                )

            
            # Log gradient norms if tracking enabled
            if track_gradients:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/layer_{name}": grad_norm,
                                "step": step,
                                "epoch": epoch + 1,
                            }
                        )

            # Run evaluation every eval_steps
            if step % eval_steps == 0:
                val_loss, val_accuracy = evaluate(model, val_loader, criterion)
                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                        "accuracy/val": val_accuracy,
                    }
                )
                logger.info(
                    f"Evaluation at step {step}: Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.2f}%"
                )
                model.train()  # Switch back to training mode

            # Log to wandb
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/train_batch": loss.item(),
                }
            )
            logger.info(f"Epoch {epoch + 1}, Step {step}: Loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        wandb.log(
            {
                "epoch": epoch + 1,
                "losses/train_epoch": avg_loss,
            }
        )
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed. Avg Loss: {avg_loss:.4f}"
        )

    wandb.finish()
    sock.close()
    logger.info("Worker training completed and connection closed.")


if __name__ == "__main__":
    main()
