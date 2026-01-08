import gc
import logging
import os
import socket
import sys
import threading
import time

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
with open("../configs/nn_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open("../configs/cluster_config_edp.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
HOST_IP = cluster_config["host_ip"][HOSTNAME]
PORT = cluster_config["port"]
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1
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

        if command == "parameter_server_reduce":
            recv_step = payload["step"]
            rank = payload["rank"]
            grads = payload["grads"]
            worker_version = payload["model_version"]
            
            logger.info(
                f"Received gradients from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
            )
            
            # Store all gradients with rank, step, and version
            with lock:
                workers_grads_received[(rank, recv_step, worker_version)] = grads
                
            gradients_event.set()
            
            logger.info(f"Gradients stored successfully for worker {rank} at step {recv_step}")
        
        elif command == 'pull_weights':
            worker_version = payload  # payload is the worker's current model version
            logger.info(f"Worker {addr} requested weights (worker version: {worker_version}, server version: {model_version})")
            
            weights = get_weights(model)
            send_message(conn, (weights, model_version))
            
            logger.info(f"Weights sent to worker {addr}")
            
        elif command == 'disconnect':
            logger.info(f"Worker {addr} requested disconnection.")
            break
        
    conn.close()
    # Remove disconnected worker
    with lock:
        workers.pop(addr, None)

def parameter_server_reduce(
    leader_grads: dict[str, torch.Tensor],
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

    if leader_grads is not None:
        for name, leader_grad in leader_grads.items():
            if name in grads_reduced:
                grads_reduced[name] = grads_reduced[name] + (leader_grad / num_workers_connected)
                
    return grads_reduced


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
    
    # Initialize W&B
    wandb.init(
        project="smolcluster",
        name=f"MNIST-training_lr_{nn_config['learning_rate']}_bsz_{nn_config['batch_size']}",
        config=nn_config,
    )

    model_summary = str(
        torchinfo.summary(model, input_size=(batch_size, 784), device=get_device())
    )
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Start accepting worker connections in background
    def accept_workers():
        while True:
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
            except Exception as e:
                logger.error(f"Error accepting worker: {e}")
    
    # Start worker acceptance thread
    threading.Thread(target=accept_workers, daemon=True).start()
    logger.info("Worker acceptance thread started")
    
    # Give workers a moment to connect
    time.sleep(2)

    optimizer = torch.optim.SGD(model.parameters(), lr=nn_config["learning_rate"])

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
        
        # Collect any available worker gradients (non-blocking with small timeout)
        gradients_event.wait(timeout=0.01)  # Very short wait
        gradients_event.clear()
        
        with lock:
            grads_copy = dict(workers_grads_received)
            workers_grads_received.clear()
        
        if grads_copy:
            logger.info(f"Step {step}: Collected {len(grads_copy)} worker gradient(s)")
            
            # Apply worker gradients with staleness scaling
            for (rank, recv_step, worker_version), grads in grads_copy.items():
                staleness = model_version - worker_version
                scale = 1.0 / (1.0 + staleness)
                
                logger.info(f"Applying worker {rank} grads from step {recv_step} (staleness: {staleness}, scale: {scale:.3f})")
                
                optimizer.zero_grad()
                scaled_grads = {k: v * scale for k, v in grads.items()}
                set_gradients(scaled_grads, model)
                optimizer.step()
                
                with lock:
                    model_version += 1
                
            del grads_copy, scaled_grads
            gc.collect()
        
        # Apply leader gradients
        optimizer.zero_grad()
        set_gradients(leader_grads, model)
        optimizer.step()
        
        with lock:
            model_version += 1
        
        del leader_grads
        gc.collect()
        
        logger.info(f"Step {step}: Updated to model version {model_version}")
        
        
        data = data.to(get_device())
        target = target.to(get_device())
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        total_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}, Step: {step}: Loss = {loss.item():.4f}, Running Avg = {total_loss/(step+1):.4f}")
        
        if step % 50 == 0:
            wandb.log(
                {
                    "step": step,
                    "losses/step_loss": loss.item(),
                }
            )
            
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
            
            logger.info(f"Evaluating model at step {step}...")
            
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            wandb.log(
                {
                    "step": step,
                    "losses/val": val_loss,
                    "accuracy/val": val_acc,
                }
            )

    logger.info(f"Training completed. Total steps: {step + 1}, Final model version: {model_version}")
    wandb.finish()
    sock.close()


if __name__ == "__main__":
    
    main()
