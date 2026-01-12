import gc
import logging
import os
import socket
import sys
import threading
from collections import defaultdict
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
    set_weights,
)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device

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
with open("../../configs/nn_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open("../../configs/cluster_config_syncps.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
HOST_IP = "0.0.0.0"
PORT = cluster_config["port"]
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1
TIMEOUT = cluster_config["timeout"]

RANK = 0
batch_size = nn_config["batch_size"]
eval_steps = nn_config["eval_steps"]
num_epochs = nn_config["num_epochs"]
track_gradients = nn_config.get("track_gradients", False)
criterion = torch.nn.CrossEntropyLoss()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[LEADER]")
logger.info(f"Server will bind to IP: {HOST_IP}, Port: {PORT}")

step_event = threading.Event()
lock = threading.Lock()

workers = {}
grads_received = defaultdict(dict)


def load_data(
    batch_size: int, WORLD_SIZE: int, SEED: int, rank: int
) -> tuple[DataLoader, DataLoader]:
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
    train_data = torch.utils.data.Subset(trainset, batch_indices[rank])
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
    accuracy = 100 * (correct / total)
    return avg_loss, accuracy


def compute_leader_gradients(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.train()
    data, target = data.to(get_device()), target.to(get_device())
    output = model(data.view(data.size(0), -1))
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    if nn_config.get("gradient_clipping", {}).get("enabled", False):
        max_norm = nn_config["gradient_clipping"].get("max_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    grads = get_gradients(model)
    return loss, grads


def handle_worker(conn: socket.SocketType, addr: tuple[str, int]) -> None:
    logger.info(f"Handling worker at {addr}")

    while True:
        try:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                logger.info(f"Worker {addr} closed connection")
                break

            # Unpack the message tuple
            command, recv_step, rank, grads = message

            logger.info(
                f"Received gradients from worker {addr} with ID {rank} for batch {recv_step}"
            )

            if command == "parameter_server_reduce":
                logger.info(
                    f"Storing gradients from worker {rank} for batch {recv_step}"
                )
                with lock:
                    curr_step = recv_step
                    grads_received[curr_step][rank] = grads
                step_event.set()
            # Add handling for other commands if needed, e.g., 'disconnect'
        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break

    logger.info(f"Worker {addr} disconnected")
    conn.close()


def parameter_server_reduce(
    grads_dict: dict[int, dict[str, torch.Tensor]], num_workers_connected: int
) -> dict[str, torch.Tensor]:
    # worker_reduced = {}
    grads_reduced = {}
    # leader_reduced = {}
    for worker_id in list(grads_dict):
        # if worker_id == RANK:
        #     continue

        for name, worker_grads in grads_dict[worker_id].items():
            grads_reduced[name] = grads_reduced.get(name, 0.0) + (
                worker_grads / num_workers_connected
            )

    return grads_reduced


model = SimpleMNISTModel(
    input_dim=nn_config["model"]["input_dim"],
    hidden=nn_config["model"]["hidden"],
    out=nn_config["model"]["out"],
)
model = model.to(get_device())
logger.info(f"Model initialized on device: {get_device()}")


train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, rank=RANK)


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
    # Initialize W&B
    wandb.init(
        project="smolcluster",
        name=f"SyncPS-server-{HOSTNAME}_lr{nn_config['learning_rate']}_bs{nn_config['batch_size']}_workers{NUM_WORKERS}",
        config={
            **nn_config,
            "server_hostname": HOSTNAME,
            "num_workers": NUM_WORKERS,
            "mode": "synchronous_ps",
        },
    )

    model_summary = str(
        torchinfo.summary(model, input_size=(batch_size, 784), device=get_device())
    )
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Accept connections and wait for registration
    registered_workers = {}  # rank -> socket
    while len(registered_workers) < NUM_WORKERS:
        client_socket, client_address = sock.accept()
        logger.info(f"Accepted connection from {client_address}")

        # Wait for registration message
        try:
            message = receive_message(client_socket)
            if message is None:
                logger.warning(
                    f"Connection from {client_address} closed before registration"
                )
                client_socket.close()
                continue

            command, rank = message
            if command == "register":
                logger.info(f"Worker {rank} registered from {client_address}")
                registered_workers[rank] = client_socket
                workers[client_address] = client_socket
                threading.Thread(
                    target=handle_worker, args=(client_socket, client_address)
                ).start()
            else:
                logger.warning(f"Unexpected message from {client_address}: {command}")
                client_socket.close()
        except Exception as e:
            logger.error(f"Error during registration from {client_address}: {e}")
            client_socket.close()
            continue

    logger.info("All workers connected. Starting training...")

    for worker_socket in registered_workers.values():
        send_message(worker_socket, "start_training")

    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            leader_loss, leader_grads = compute_leader_gradients(
                model, data, target, criterion, optimizer
            )
            grads_received[step][RANK] = leader_grads

            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/leader_step": leader_loss.item(),
                }
            )
           
            while True:
                with lock:
                    curr_workers_len = len(grads_received[step])

                logger.info(
                    f"Epoch {epoch + 1}, Step: {step}, Batch {batch_idx}: Received gradients from {curr_workers_len}/{WORLD_SIZE} participants."
                )
                if curr_workers_len < NUM_WORKERS:
                    logger.info(f"Waiting for more gradients for step {step}...")
                    step_event.wait()
                    step_event.clear()

                else:
                    break

            if len(grads_received[step]) != 0:
                grads_reduced = parameter_server_reduce(
                    grads_received[step], len(grads_received[step])
                )

                # Send gradients to workers
                for _worker_addr, worker_socket in workers.items():
                    send_message(
                        worker_socket, ("averaged_gradients", step, grads_reduced)
                    )

                set_weights(grads_reduced, model)

                optimizer.step()
                grads_received.pop(step, None)
                del grads_reduced, leader_grads
                gc.collect()

            else:
                logger.warning(
                    f"No gradients received for step {step}. Skipping grad update."
                )
                del leader_grads
                
            data = data.to(get_device())
            target = target.to(get_device())
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            total_loss += loss.item()

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

            # Log training metrics
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/train_step": loss.item(),
                    "lr": nn_config["learning_rate"],
                    "batch_size": nn_config["batch_size"],
                }
            )

            if step % eval_steps == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion)

                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                        "accuracy/val": val_acc,
                    }
                )
                logger.info(f"Step {step}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

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
    
    logger.info("Training completed successfully!")
    wandb.finish()
    sock.close()


if __name__ == "__main__":
    main()
