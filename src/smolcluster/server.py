
from collections import defaultdict
import socket
import yaml
import torchvision
from utils.common_utils import send_message, receive_message, get_gradients, set_weights
from utils.data import get_data_indices
from utils.device import get_device
from models.SimpleNN import SimpleMNISTModel
import torch
import threading



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

local_rank = input("Enter worker ID (0 to {}): ".format(NUM_WORKERS - 1))
local_rank = int(local_rank)

RANK = local_rank
batch_size = nn_config['batch_size']

step_event = threading.Event()
lock = threading.Lock()

workers = {}
grads_received = defaultdict(dict)


def load_data(batch_size, WORLD_SIZE, SEED, RANK):
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
    accuracy = 100 * (correct / total)
    return avg_loss, accuracy


def compute_leader_gradients(model: torch.nn.Module, data, target, criterion):
    model.train()
    data, target = data.to(get_device()), target.to(get_device())
    model.zero_grad()
    output = model(data.view(data.size(0), -1))
    loss = criterion(output, target)
    loss.backward()
    grads = get_gradients(model)
    return grads


def handle_worker(conn, addr):

    


model = SimpleMNISTModel(input_dim=nn_config['model']['input_dim'], hidden=nn_config['model']['hidden'], out=nn_config['model']['out'])
model = model.to(get_device())
print("Model initialized on device:", get_device())


train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, RANK)


print("Data ready. Train size: {}, Test size: {}".format(len(train_loader), len(val_loader)))


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST_IP, PORT))

print("Worker connected to server at {}:{}".format(HOST_IP, PORT))

# Bind the socket to the host and port
sock.bind((HOST_IP, PORT))

# Listen for incoming connections
sock.listen(5)
print(f"Server listening on {HOST_IP}:{PORT}")


def main():
    # Accept connections
    while len(workers) <= NUM_WORKERS:
        
        client_socket, client_address = sock.accept()
        print(f"Accepted connection from {client_address}")
        # Handle the connection (you can add more logic here)
        workers[client_address] = client_socket
        threading.Thread(target=handle_worker, args=(client_socket, client_address)).start()
        
        
    print("All workers connected. Starting training...")
    
    num_epochs = nn_config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config['learning_rate'])
    
    for epoch in range(num_epochs):
        model.train()
        if RANK == 0:
            total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            leader_grads = compute_leader_gradients(model, data, target, criterion)
            grads[batch_idx][]
            
    client_socket.close()

