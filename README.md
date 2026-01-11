# SmolCluster

A distributed training framework for MNIST classification using PyTorch and socket-based communication. This project demonstrates elastic distributed training concepts with a central server coordinating gradient averaging across heterogeneous worker nodes (Mac minis, Raspberry Pis, MacBook).

## Features

- **Elastic Distributed Training**: Coordinate training across multiple workers with different compute capabilities
- **Hybrid Network Topology**: Thunderbolt fabric for inter-Mac communication + Ethernet for Pi edge nodes
- **Socket Communication**: TCP socket-based communication with retry logic and connection resilience
- **W&B Integration**: Automatic logging of training metrics with detailed run naming (hostname, LR, batch size)
- **Configurable**: YAML-based configuration for easy experimentation
- **MNIST Dataset**: Built-in support for MNIST handwritten digit recognition
- **Gradient Tracking**: Optional gradient norm logging for analysis
- **Quantization Support**: Optional gradient quantization for bandwidth-limited links

## Network Architecture

**Thunderbolt Fabric** (High-speed inter-Mac communication):
```
Mac mini 1 (SERVER) — 10.10.0.1  ─┐
Mac mini 2 (WORKER) — 10.10.0.2  ─┼─ Thunderbolt Bridge
Mac mini 3 (WORKER) — 10.10.0.3  ─┘
```

**Ethernet Edge Links** (Pi connectivity):
```
Pi 5 (192.168.50.2) ──── Mac mini 1 (192.168.50.1)  [Server path]
Pi 4 (192.168.51.4) ──── Mac mini 3 (192.168.51.2)  [Worker gateway]
```

**Key Design Principle**: One subnet per physical link. Pis route to Thunderbolt network via their connected Mac.

## Prerequisites

- Python 3.9.6+ (tested with socket communication in tmux environments)
- uv package manager (install from [astral.sh/uv](https://astral.sh/uv))

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YuvrajSingh-mist/smolcluster.git
   cd smolcluster
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required packages including PyTorch, torchvision, wandb, and others.

## Configuration

The project uses two main configuration files located in `src/smolcluster/configs/`:

### cluster_config_edp.yaml (Elastic Distributed Training)
```yaml
host_ip:
  mini1: "10.10.0.1"      # Server on Thunderbolt network
  mini2: "10.10.0.2"      # Worker on Thunderbolt network  
  mini3: "10.10.0.3"      # Worker on Thunderbolt network
  pi4: "192.168.51.4"     # Worker via Ethernet (routes through mini3)
  pi5: "192.168.50.2"     # Worker via Ethernet (routes through mini1)
  macbook: "172.18.3.80"  # Worker on WiFi

port: 65432
num_workers: 3              # Number of worker nodes
workers: [pi4, pi5, macbook]  # Worker hostnames
server: mini1                 # Server hostname
worker_update_interval: 50    # Steps between weight pulls
use_quantization: false       # Enable gradient quantization
seed: 42
```

**Configuration Details:**
- `host_ip`: Maps hostnames to their network addresses. Server uses Thunderbolt IP (10.10.0.1).
- `port`: TCP port for socket communication (65432 by default).
- `num_workers`: Number of worker nodes expected.
- `workers`: List of worker hostnames (must match keys in `host_ip`).
- `server`: Server hostname (must match key in `host_ip`).
- `worker_update_interval`: How often workers pull latest weights from server (in training steps).
- `use_quantization`: Enable 8-bit quantization for gradient communication (reduces bandwidth).
- `seed`: Random seed for reproducible data partitioning.

### nn_config.yaml
```yaml
batch_size: 32
learning_rate: 0.001
num_epochs: 2
eval_steps: 200        # Evaluate every N steps
track_gradients: true  # Log gradient norms to W&B
model:
  type: SimpleNN
  input_dim: 784
  hidden: 128
  out: 10
dataset:
  name: MNIST
```

## Automated Cluster Launch

The `launch.sh` script automates the process of starting distributed training across multiple Mac mini nodes via SSH. It handles SSH connectivity checks, remote environment setup, and tmux session management.


### How launch.sh Works

1. **Connectivity Checks**: Verifies SSH access, tmux installation, and uv availability on all nodes
2. **Environment Setup**: Ensures Python 3.9.6 virtual environment exists on each node
3. **Session Cleanup**: Kills any existing tmux sessions to prevent conflicts
4. **Server Launch**: Starts the central server on the first node (mini1) in a tmux session
5. **Worker Launch**: Starts worker processes on subsequent nodes (mini2, mini3, etc.)
6. **Logging**: All output is logged to `~/session_name.log` on each node

### Usage

```bash
# Dry run to see what would be executed
./launch.sh --dry-run

# Actually launch the cluster
./launch.sh
```

### Monitoring and Troubleshooting

After launching, you can:

```bash
# Check tmux sessions on server node
ssh mini1 'tmux ls'

# Attach to server session
ssh mini1 'tmux attach -t server'

# Attach to worker session
ssh mini2 'tmux attach -t worker1'

# View logs
ssh mini1 'tail -f ~/server.log'
ssh mini2 'tail -f ~/worker1.log'
```
## Usage

### 1. Start the Server

Open a terminal and start the central server:

```bash
cd src/smolcluster/DDP/SimpleAllReduce
../../../../.venv/bin/python server.py
```

The server will:
- Initialize the model
- Load MNIST data
- Wait for worker connections
- Coordinate training and gradient averaging

### 2. Start Workers

For each worker, open a new terminal and run:

```bash
cd src/smolcluster/DDP/SimpleAllReduce
../../../../.venv/bin/python worker.py
```

When prompted, enter the worker ID (1, 2, ..., num_workers).

Each worker will:
- Connect to the server
- Load its portion of the MNIST data
- Compute gradients on local data
- Send gradients to server for averaging
- Receive averaged gradients and update local model

### 3. Monitor Training

- **Console Logs**: View real-time training progress in each terminal
- **W&B Dashboard**: Visit [wandb.ai](https://wandb.ai) to see detailed metrics, losses, and gradient norms
- **Project Name**: `smolcluster`

## Project Structure

```
smolcluster/
├── src/smolcluster/
│   ├── DDP/
│   │   ├── SimpleAllReduce/       # Distributed training implementation
│   │   │   ├── server.py       # Central server coordinating training
│   │   │   └── worker.py       # Worker node for distributed training
│   ├── main.py            # (Optional) Standalone training script
│   ├── models/
│   │   ├── __init__.py
│   │   └── SimpleNN.py    # Simple neural network model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── common_utils.py # Gradient ops, messaging
│   │   ├── data.py        # Data partitioning utilities
│   │   └── device.py      # Device detection
│   ├── data/
│   │   └── MNIST/         # MNIST dataset (auto-downloaded)
│   │       └── raw/       # Raw MNIST files
│   └── configs/
│       ├── cluster_config_ddp.yaml  # DDP cluster network settings
│       ├── cluster_config_edp.yaml  # EDP cluster network settings
│       └── nn_config.yaml       # Model/training settings
│   └── docs/
│       └── setup_cluster.md     # Mac mini cluster setup guide
├── launch.sh              # Automated cluster launch script
├── pyproject.toml         # Project dependencies and config
├── README.md             # This file
└── .gitignore
```

## How It Works

1. **Data Partitioning**: MNIST training data is split across workers using a deterministic seed-based approach
2. **Gradient Computation**: Each worker computes gradients on its local data partition
3. **Gradient Averaging**: Server collects gradients from all workers, computes average, and redistributes
4. **Model Updates**: Each node updates its model using the averaged gradients
5. **Synchronization**: Process repeats for each batch across epochs


## Setting Up a Mac Mini Cluster
For setup instructions for creating a Mac mini cluster using Thunderbolt and SSH, refer to the [setup_cluster.md](src/smolcluster/docs/setup_cluster.md) guide.

