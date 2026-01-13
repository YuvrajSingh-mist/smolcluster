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
    get_weights,
    receive_message,
    send_message,
)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device
from smolcluster.utils.quantization import (
    calculate_compression_ratio,
    dequantize_model_weights,
    quantize_model_weights,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[WORKER]")

# Global variables for model versioning
model_version = 0
recv_model_version = -1


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


def run_edp_worker(
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
    Run EDP worker training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        worker_rank: Worker rank (0-indexed)
        hostname: Worker hostname
        device: Device to run on
        criterion: Loss criterion
        host_ip: Server IP address
        port: Server port
    """
    global model_version, recv_model_version
    
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config.get("track_gradients", False)
    learning_rate = config["learning_rate"]
    use_quantization = cluster_config.get("use_quantization", True)
    worker_update_interval = cluster_config.get("worker_update_interval", 5)

    # Initialize W&B for worker
    wandb.init(
        project="smolcluster",
        name=f"worker-{hostname}_rank{worker_rank}_lr{learning_rate}_bs{batch_size}",
        config={
            **config,
            "worker_rank": worker_rank,
            "worker_hostname": hostname,
            "server_hostname": cluster_config["server"],
            "worker_update_interval": worker_update_interval,
        },
    )
    logger.info(f"Worker {worker_rank} wandb initialized")

    # Connect to server with retry logic
    sock = connect_to_server(host_ip, port)

    # Register with the server
    logger.info(f"Registering as worker {worker_rank} with server...")
    send_message(sock, ("register", worker_rank))

    logger.info(
        f"Data loaders ready. Train size: {len(train_loader)}, Test size: {len(val_loader)}"
    )

    total_steps = num_epochs * len(train_loader)

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
        # optimizer.zero_grad()
        data, target = data.to(get_device()), target.to(get_device())

        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if config.get("gradient_clipping", {}).get("enabled", False):
            max_norm = config["gradient_clipping"].get("max_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()  # Local SGD: workers apply updates independently

        # Send locally-updated weights for Polyak averaging on server
        weights = get_weights(model)

        if use_quantization:
            quantized_weights = quantize_model_weights(weights)

            # Log compression ratio on first step
            if step == 0:
                comp_info = calculate_compression_ratio(weights, quantized_weights)
                logger.info(
                    f"Quantization: {comp_info['original_mb']:.2f}MB → {comp_info['compressed_mb']:.2f}MB (ratio: {comp_info['ratio']:.2f}x)"
                )

            logger.info(
                "Local forward and backward pass done. Sending quantized model weights to server."
            )
            send_message(
                sock,
                (
                    "polyark_averaging",
                    {
                        "step": step,
                        "rank": worker_rank,
                        "quantized_weights": quantized_weights,
                        "model_version": model_version,
                    },
                ),
            )
            logger.info("Quantized model weights sent to server.")
        else:
            logger.info(
                "Local forward and backward pass done. Sending model weights to server."
            )
            send_message(
                sock,
                (
                    "polyark_averaging",
                    {
                        "step": step,
                        "rank": worker_rank,
                        "weights": weights,
                        "model_version": model_version,
                    },
                ),
            )
            logger.info("Model weights sent to server.")

        
        if step % worker_update_interval == 0 and step != 0:
            logger.info(f"Pulling weights from server at step {step}.")
            send_message(sock, ("pull_weights", model_version))
            sock.settimeout(1.0)  # Wait up to 1 second for weights
            try:
                weights, new_version = receive_message(sock)

                if use_quantization:
                    dequant_weights = dequantize_model_weights(
                        weights, device=get_device()
                    )
                    model.load_state_dict(dequant_weights)
                else:
                    # print(weights)
                    model.load_state_dict(weights)

                recv_model_version = new_version
                logger.info(
                    f"Updated to model version {recv_model_version} from server."
                )
            except socket.timeout:
                logger.warning("Timeout while pulling weights from server.")
            except BlockingIOError:
                logger.error(
                    "non-blocking socket error while pulling weights from server."
                )
            finally:
                sock.settimeout(None)  # Restore blocking socket

            if track_gradients:
                logger.info("Tracking gradients in wandb...")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        logger.info(f"Logging gradients for layer: {name}")
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/layer_{name}": grad_norm,
                                "step": step,
                                "epoch": epoch,
                            }
                        )
                logger.info("Gradient tracking complete.")

        # Update local model version if received new weights
        if recv_model_version != -1 and recv_model_version != model_version:
            model_version = recv_model_version
            logger.info(f"Updated local model version to {model_version}.")

        # Run evaluation every eval_steps
        if step % eval_steps == 0:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "losses/val": val_loss,
                    "accuracy/val": val_accuracy,
                    "losses/train_batch": loss.item(),
                }
            )
            logger.info(
                f"Evaluation at step {step}: Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.2f}%"
            )
        else:
            # Log training loss only
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "losses/train_batch": loss.item(),
                }
            )

        logger.info(f"Epoch: {epoch} , Step {step}/{total_steps} completed.")

    # Finish wandb tracking
    wandb.finish()
    logger.info(f"Worker {worker_rank} wandb tracking finished.")

    send_message(sock, ("disconnect", worker_rank))
    sock.close()
    logger.info(f"Training complete. Worker {worker_rank} disconnected.")


def main():
    """Legacy main function for backward compatibility."""
    global model_version, recv_model_version
    
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

    with open(CONFIG_DIR / "cluster_config_edp.yaml") as f:
        cluster_config = yaml.safe_load(f)

    # Extract values with defaults
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
    PORT = cluster_config["port"]
    
    criterion = torch.nn.CrossEntropyLoss()
    
    model = SimpleMNISTModel(
        input_dim=nn_config["model"]["input_dim"],
        hidden=nn_config["model"]["hidden"],
        out=nn_config["model"]["out"],
    )
    model = model.to(get_device())
    logger.info(f"Model initialized on device: {get_device()}")

    train_loader, val_loader = load_data(nn_config["batch_size"], WORLD_SIZE, SEED, local_rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])
    
    run_edp_worker(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=nn_config,
        cluster_config=cluster_config,
        worker_rank=local_rank,
        hostname=HOSTNAME,
        device=get_device(),
        criterion=criterion,
        host_ip=HOST_IP,
        port=PORT,
    )


if __name__ == "__main__":
    main()
