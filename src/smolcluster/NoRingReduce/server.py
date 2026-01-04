import gc
import logging
import socket
import threading
import time
from collections import defaultdict

import torch
import torchinfo
import torchvision
import wandb
import yaml
from smolcluster.models.SimpleNN import SimpleMNISTModel
from torch.utils.data import DataLoader
from smolcluster.utils.common_utils import get_gradients, receive_message, send_message, set_weights
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device

# Load configs
with open("../configs/nn_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open("../configs/cluster_config.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
HOST_IP = cluster_config.get("host_ip") or socket.gethostbyname(socket.gethostname())
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
logger = logging.getLogger("Server")

step_event = threading.Event()
lock = threading.Lock()

workers = {}
grads_received = defaultdict(dict)


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
    logger.info(f"Handling worker at {addr}")

    while True:
        try:
            command, recv_step, rank, grads = receive_message(conn)

            logger.info(
                f"Received gradients from worker {addr} with ID {rank} for batch {recv_step}"
            )

            if command == "all_reduce":
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


def all_reduce(
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


train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, RANK)


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
        name=f"MNIST-training_lr_{nn_config['learning_rate']}_bsz_{nn_config['batch_size']}",
        config=nn_config,
    )

    if RANK == 0:
        model_summary = str(
            torchinfo.summary(model, input_size=(batch_size, 784), device=get_device())
        )
        logger.info("Model Summary:")
        logger.info(model_summary)
        wandb.log({"model_structure": model_summary})

    # Accept connections
    while len(workers) < NUM_WORKERS:
        client_socket, client_address = sock.accept()
        logger.info(f"Accepted connection from {client_address}")
        # Handle the connection (you can add more logic here)
        workers[client_address] = client_socket
        threading.Thread(
            target=handle_worker, args=(client_socket, client_address)
        ).start()

    logger.info("All workers connected. Starting training...")

    for worker_socket in workers.values():
        send_message(worker_socket, "start_training")

    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        model.train()
        if RANK == 0:
            total_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            leader_grads = compute_leader_gradients(
                model, data, target, criterion, optimizer
            )
            grads_received[step][RANK] = leader_grads

            start_time = time.time()

            while True:
                with lock:
                    curr_workers_len = len(grads_received[step])

                logger.info(
                    f"Epoch {epoch + 1}, Step: {step}, Batch {batch_idx}: Received gradients from {curr_workers_len}/{WORLD_SIZE} participants."
                )
                if curr_workers_len < NUM_WORKERS:
                    logger.info(f"Waiting for more gradients for step {step}...")
                    curr_time = time.time()

                    if curr_time - start_time >= TIMEOUT:
                        logger.warning(
                            f"Timeout waiting for gradients for step {step}. Proceeding with available gradients."
                        )
                        break
                    else:
                        step_event.wait(timeout=TIMEOUT)
                        step_event.clear()

                else:
                    break

            if len(grads_received[step]) != 0:
                grads_reduced = all_reduce(
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
                    f"No gradients received for step {step}. Skipping weight update."
                )
            if RANK == 0:
                data = data.to(get_device())
                target = target.to(get_device())
                output = model(data.view(data.size(0), -1))
                loss = criterion(output, target)
                total_loss += loss.item()

                if track_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/layer_{name}": grad_norm,
                                    "step": step,
                                }
                            )

                wandb.log(
                    {
                        "step": step,
                        "lr": nn_config["learning_rate"],
                        "batch_size": nn_config["batch_size"],
                    }
                )

                if step % eval_steps == 0:
                    val_loss, val_acc = evaluate(model, val_loader, criterion)

                    wandb.log(
                        {
                            "step": step,
                            "losses/val": val_loss,
                            "accuracy/val": val_acc,
                        }
                    )

        avg_loss = total_loss / len(train_loader)

        wandb.log(
            {
                "step": step,
                "losses/train": avg_loss,
            }
        )

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Step: {step}/{num_epochs * len(train_loader)} completed."
        )

    wandb.finish()
    client_socket.close()


if __name__ == "__main__":
    main()
