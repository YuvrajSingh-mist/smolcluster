import socket
import yaml
import torchvision
from utils.common_utils import send_message, receive_message, get_gradients, set_weights
from utils.data import get_data_indices
from utils.device import get_device
from models.SimpleNN import SimpleMNISTModel
import torch

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
    accuracy = 100 * correct / total
    return avg_loss, accuracy




sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST_IP, PORT))

print(f"[Worker {RANK}] connected to server at {HOST_IP}:{PORT}")


def main():

    model = SimpleMNISTModel(input_dim=nn_config['model']['input_dim'], hidden=nn_config['model']['hidden'], out=nn_config['model']['out'])
    model = model.to(get_device())
    print(f"[Worker {RANK}] Model initialized on device: {get_device()}")

    train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, RANK)
    print(f"[Worker {RANK}] Data loaders ready. Train size: {len(train_loader)}, Test size: {len(val_loader)}")
    
    
    num_epochs = nn_config['num_epochs']
    eval_step = nn_config['eval_step']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config['learning_rate'])
    
    while True:
        recv_command = receive_message(sock)
        print(recv_command)
        if recv_command == 'start_training':
            print(f"[Worker {RANK}] Received start_training command from server.")
            break
    
    
    for epoch in range(num_epochs):
        model.train()
        if RANK == 0:
            total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            data, target = data.to(get_device()), target.to(get_device())
            
            
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            
            loss.backward()
            grads = get_gradients(model)

            # Send gradients to server and receive updated weights
            send_message(sock, ('all_reduce', batch_idx, RANK, grads))
            
            data_recv = receive_message(sock)
            
            # print(data_recv, f"[Worker {RANK}] received data from server")
            command, recv_step, updated_grads = data_recv 
            
            # print(command, recv_step, updated_grads)
            assert recv_step == batch_idx, f"[Worker {RANK}] Mismatched step from server"
            
            if recv_step > batch_idx:
                batch_idx = recv_step
            # else:
                
            if command == 'averaged_gradients':
                set_weights(updated_grads, model)
            
            optimizer.step()
            
            print(f"[Worker {RANK}] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} completed.")
            if RANK == 0:
                total_loss += loss.item()
            
        if RANK == 0 and (batch_idx + 1) % eval_step == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            print(f'Epoch {epoch+1}, Train Loss: {total_loss / (len(train_loader)):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
if __name__ == "__main__":
    main()