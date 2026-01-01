from collections import defaultdict
import time
import socket
from sympy import reduced
import yaml
import torchvision
from utils.common_utils import send_message, receive_message, get_gradients, set_weights
from utils.data import get_data_indices
from utils.device import get_device
from models.SimpleNN import SimpleMNISTModel
import torch
import threading
from typing import Dict, Tuple
from torch.utils.data import DataLoader



# Load configs
with open('configs/nn_config.yaml', 'r') as f:
    nn_config = yaml.safe_load(f)

with open('configs/cluster_config.yaml', 'r') as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
HOST_IP = cluster_config.get('host_ip') or socket.gethostbyname(socket.gethostname())
PORT = cluster_config['port']
NUM_WORKERS = cluster_config['num_workers']
SEED = cluster_config.get('seed', 42)
WORLD_SIZE = NUM_WORKERS
TIMEOUT = cluster_config['timeout']

RANK = 0
batch_size = nn_config['batch_size']

step_event = threading.Event()
lock = threading.Lock()

workers = {}
grads_received = defaultdict(dict)


def load_data(batch_size: int, WORLD_SIZE: int, SEED: int, RANK: int) -> Tuple[DataLoader, DataLoader]:
    # load MNIST
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    data = torchvision.datasets.MNIST('.', download=True, transform=transforms)
    lendata = len(data)
    trainset, testset = torch.utils.data.random_split(data, [int(0.9 * lendata), lendata - int(0.9 * lendata)])
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    batch_indices = get_data_indices(len(trainset), WORLD_SIZE, SEED)
    train_data = torch.utils.data.Subset(trainset, batch_indices[RANK])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module) -> Tuple[float, float]:
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


def compute_leader_gradients(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, criterion: torch.nn.Module) -> Dict[str, torch.Tensor]:
    model.train()
    data, target = data.to(get_device()), target.to(get_device())
    model.zero_grad()
    output = model(data.view(data.size(0), -1))
    loss = criterion(output, target)
    loss.backward()
    grads = get_gradients(model)
    return grads


def handle_worker(conn: socket.SocketType, addr: Tuple[str, int]) -> None:
    print(f"[Leader] Handling worker at {addr}")
    
    while True:
        try:
            command, recv_step, rank, grads = receive_message(conn)
            
            print(f"[Leader] Received gradients from worker {addr} with ID {rank} for batch {recv_step}") 
            
            if command == 'all_reduce':
                print(f"[Leader] Storing gradients from worker {rank} for batch {recv_step}")
                with lock:
                    curr_step = recv_step
                    grads_received[curr_step][rank] = grads
                step_event.set()
            # Add handling for other commands if needed, e.g., 'disconnect'
        except Exception as e:
            print(f"[Leader] Error handling worker {addr}: {e}")
            break
    
    print(f"[Leader] Worker {addr} disconnected")
    conn.close()

def all_reduce(grads_dict: Dict[int, Dict[str, torch.Tensor]], num_workers_connected: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    
    worker_reduced = {}
    leader_reduced = {}
    for worker_id in grads_dict:
        
        if worker_id == RANK:
            continue
        
        for name, worker_grads in grads_dict[worker_id].items():
            
            worker_reduced[name] = worker_reduced.get(name, 0.0) + (worker_grads / num_workers_connected)
            
        
        grads_dict[worker_id].pop(name)
    
    for name in grads_dict[RANK]:
        leader_reduced[name] = leader_reduced.get(name, 0.0) + (grads_dict[RANK][name] / num_workers_connected)
        
    return leader_reduced, worker_reduced
    

model = SimpleMNISTModel(input_dim=nn_config['model']['input_dim'], hidden=nn_config['model']['hidden'], out=nn_config['model']['out'])
model = model.to(get_device())
print("[Leader] Model initialized on device:", get_device())


train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, RANK)


print("[Leader] Data ready. Train size: {}, Test size: {}".format(len(train_loader), len(val_loader)))


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
sock.bind((HOST_IP, PORT))

# Listen for incoming connections
sock.listen(5)
print(f"[Leader] Server listening on {HOST_IP}:{PORT}")


def main():
    
    # Accept connections
    while len(workers) < NUM_WORKERS:
        
        client_socket, client_address = sock.accept()
        print(f"[Leader] Accepted connection from {client_address}")
        # Handle the connection (you can add more logic here)
        workers[client_address] = client_socket
        threading.Thread(target=handle_worker, args=(client_socket, client_address)).start()
        
        
    print("[Leader] All workers connected. Starting training...")
    
    for worker_socket in workers.values():
        send_message(worker_socket, 'start_training')
     
    num_epochs = nn_config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config['learning_rate'])
    
    print(f"[Leader] Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        model.train()
        if RANK == 0:
            total_loss = 0.0
        print(f"[Leader] Starting epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            
            leader_grads = compute_leader_gradients(model, data, target, criterion)
            grads_received[batch_idx][RANK] = leader_grads
            start_time = time.time()
            
           
            while True:
                
                with lock:
                    curr_workers_len = len(grads_received[batch_idx])
                
                print(f"[Leader] Epoch {epoch+1}, Batch {batch_idx}: Received gradients from {curr_workers_len}/{NUM_WORKERS} workers.")
                if curr_workers_len < NUM_WORKERS:
                    
                    print(f"[Leader] Waiting for more gradients for batch {batch_idx}...")
                    curr_time = time.time()
                    
                    if curr_time - start_time >= TIMEOUT:
                        print(f"[Leader] Timeout waiting for gradients for batch {batch_idx}. Proceeding with available gradients.")
                        break
                    else:
                        step_event.wait(timeout=TIMEOUT)
                        step_event.clear()

                else:
                    break
                
        
            if len(grads_received[batch_idx]) != 0:
                leader_reduced, worker_reduced = all_reduce(grads_received[batch_idx], len(grads_received[batch_idx]))

                # Send gradients to workers
                for worker_addr, worker_socket in workers.items():
                    send_message(worker_socket, ('averaged_gradients', batch_idx, worker_reduced))
                
                optimizer.zero_grad()
                
                set_weights(leader_reduced, model)
                
                optimizer.step()
            else:
                print(f"[Leader] No gradients received for batch {batch_idx}. Skipping weight update.")
                
            
            if RANK == 0:
                data = data.to(get_device())
                target = target.to(get_device())
                output = model(data.view(data.size(0), -1))
                loss = criterion(output, target)
                total_loss += loss.item()
                
                
        avg_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        print(f"[Leader] Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"[Leader] Epoch {epoch+1} completed.")
        
    client_socket.close()

if __name__ == "__main__":
    main()