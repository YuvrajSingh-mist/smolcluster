import gc
import logging
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
NUM_SLOW_WORKERS = len(cluster_config.get("slow_workers") or [])
NUM_FAST_WORKERS = len(cluster_config.get("fast_workers") or [])
NUM_WORKERS = NUM_SLOW_WORKERS + NUM_FAST_WORKERS
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1
FAST_WORKERS_IPS = cluster_config.get("fast_workers") or {}
SLOW_WORKERS_IPS = cluster_config.get("slow_workers") or {}
RANK = 0
FAST_WORKER_TIMEOUT = cluster_config["fast_worker_timeout"]
SLOW_WORKER_TIMEOUT = cluster_config["slow_worker_timeout"]

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

fast_step_event = threading.Event()
slow_step_event = threading.Event()
lock = threading.Lock()

model_version = 0  # Track global model version for elastic training

workers = {}
fast_workers_grads_received = {}
slow_workers_grads_received = {}
all_workers_ips_addr = {
    "fast_workers": [ip for ip in FAST_WORKERS_IPS.values()],
    "slow_workers": [ip for ip in SLOW_WORKERS_IPS.values()],
}


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
        try:
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
                    f"Received gradients from worker {addr} with ID {rank} for received step {recv_step} (worker version: {worker_version})"
                )
                
                logger.info(
                    f"Storing gradients from worker {rank} for received step {recv_step}"
                )
               
                ip_address, _port = addr
                
                if ip_address in all_workers_ips_addr["fast_workers"]:
                    with lock:
                        fast_workers_grads_received[(rank, worker_version)] = grads
                        
                    fast_step_event.set()
                    
                    logger.info(f"Gradients stored successfully for fast worker {rank} at step {recv_step}")
                
                elif ip_address in all_workers_ips_addr["slow_workers"]:
                    with lock:
                        slow_workers_grads_received[(rank, worker_version)] = grads
                     
                    slow_step_event.set()
                    logger.info(f"Gradients stored successfully for slow worker {rank} at step {recv_step}") 
                        
            # Add handling for other commands if needed, e.g., 'disconnect'
            
            elif command == "pull_weights":
                worker_version = payload  # payload is just the model_version integer
                logger.info(f"Worker {addr} requested weights (worker version: {worker_version}, current server version: {model_version})")
                
                weights = get_weights(model)
                send_message(conn, (weights, model_version))
                
                logger.info(f"Weights sent to worker {addr}")
            
            elif command == "disconnect":
                logger.info(f"Worker {addr} requested disconnection.")
                with lock:
                    workers.pop(addr, None)
                break
            
        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break

    logger.info(f"Worker {addr} disconnected")
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

    if RANK == 0:
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

    optimizer = torch.optim.SGD(model.parameters(), lr=nn_config["learning_rate"])

    logger.info(f"Starting training for {num_epochs} epochs.")
    
    train_iter = iter(train_loader)

    total_steps = num_epochs * len(train_loader)
   
    for step in range(total_steps):
        model.train()
        
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        
        data = data.to(get_device())
        target = target.to(get_device())
        
        if RANK == 0:
            total_loss = 0.0
        
        epoch = step // len(train_loader)
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        leader_grads = compute_leader_gradients(
            model, data, target, criterion, optimizer
        )
        logger.info(f"Epoch {epoch + 1}, Step: {step}: Computed leader gradients.")
        
        # Handle fast workers
        if NUM_FAST_WORKERS > 0:
            while True:
                with lock:
                    curr_workers_len_fast = len(fast_workers_grads_received)
                  
                logger.info(
                    f"Epoch {epoch + 1}, Step: {step}: Received gradients from {curr_workers_len_fast}/{NUM_FAST_WORKERS} fast participants."
                )
                
                FAST_QUORUM = max(1, int(0.7 * NUM_FAST_WORKERS))
                
                if len(fast_workers_grads_received) == 0:
                    break
                
                elif curr_workers_len_fast >= FAST_QUORUM:
                    logger.info(f"Enough fast workers. Proceeding with available gradients {len(fast_workers_grads_received)}.")
                    break
                else:
                    fast_step_event.wait(timeout=FAST_WORKER_TIMEOUT)
                    fast_step_event.clear()
        
            if len(fast_workers_grads_received) != 0:
                logger.info(f"Received gradients from fast workers for step {step}.")
                with lock:
                
                    fast_grads_copy = dict(fast_workers_grads_received)
                    fast_workers_grads_received.clear()
                    
                    fast_workers_grads_updated = {k[0]: v for k, v in fast_grads_copy.items()}
                    
                logger.info("Reducing gradients from fast workers and leader.")
                grads_reduced = parameter_server_reduce(
                    leader_grads,
                    fast_workers_grads_updated, WORLD_SIZE 
                )
                
                optimizer.zero_grad()
                set_gradients(grads_reduced, model)
                optimizer.step()
                
                logger.info(f"Updated model with reduced gradients for step {step}")
                
                with lock:
                    model_version += 1

                logger.info(f"Updated to model version {model_version}")
                
                
                del grads_reduced, leader_grads, fast_grads_copy, fast_workers_grads_updated
                gc.collect()

                logger.info("Latest weights pull signal sent to the workers. Waiting for slow workers gradients...")
            
            else:
                logger.info("No fast worker gradients available after filtering, using only leader gradients.")
                optimizer.zero_grad()
                set_gradients(leader_grads, model)
                optimizer.step()
                with lock:
                    model_version += 1
                del leader_grads
                gc.collect()
                logger.info(f"Updated to model version {model_version} using only leader gradients.")
           
        
        # Handle slow workers
        if NUM_SLOW_WORKERS > 0 and len(workers) > 0:
            while True:
                with lock:
                    curr_workers_len_slow = len(slow_workers_grads_received)
                logger.info(
                    f"Epoch {epoch + 1}, Step: {step}: Received gradients from {curr_workers_len_slow}/{NUM_SLOW_WORKERS} slow participants."
                )
                if curr_workers_len_slow < NUM_SLOW_WORKERS:
                    if not slow_step_event.wait(timeout=SLOW_WORKER_TIMEOUT):
                        logger.warning(
                            f"Timeout waiting for gradients for step {step} for slow workers. Proceeding with available gradients {len(slow_workers_grads_received)}."
                        )
                        break
                    slow_step_event.clear()
                else:
                    break
        
            if len(slow_workers_grads_received) != 0:
                with lock:
                    slow_grads_copy = slow_workers_grads_received.copy()
                    slow_workers_grads_received.clear()

                    
                logger.info(f"Updating model with {len(slow_grads_copy)} slow worker gradients using elastic SGD")
                
                for (rank, worker_version), grads in slow_grads_copy.items():
                
                    
                    staleness = model_version - worker_version
                    scale = 1.0 / (1.0 + staleness)
                    
                    logger.info(f"Applying slow worker rank {rank} grads (staleness: {staleness}, scale: {scale:.3f})")
                    
                    # Scale gradients by staleness
                    optimizer.zero_grad()
                    scaled_grads = {k: v * scale for k, v in grads.items()}
                    
                    set_gradients(scaled_grads, model)
                    optimizer.step()
                    
                    with lock:
                        model_version += 1
                    logger.info(f"Updated to model version {model_version} after applying slow worker {rank} gradients")
                
                del slow_grads_copy, scaled_grads
                gc.collect()
        
            
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
        f"Epoch {epoch + 1}/{num_epochs}, Step: {step}/{num_epochs * len(train_loader) * NUM_WORKERS} completed."
    )

    wandb.finish()
    client_socket.close()


if __name__ == "__main__":
    
    main()
