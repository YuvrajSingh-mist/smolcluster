# SmolCluster

A simple distributed training framework for MNIST classification using PyTorch and socket-based communication. This project demonstrates federated learning concepts with a central server coordinating gradient averaging across multiple worker nodes.

## Features

- **Distributed Training**: Coordinate training across multiple workers using gradient averaging
- **Socket Communication**: Simple TCP socket-based communication between server and workers
- **W&B Integration**: Automatic logging of training metrics and model performance
- **Configurable**: YAML-based configuration for easy experimentation
- **MNIST Dataset**: Built-in support for MNIST handwritten digit recognition
- **Gradient Tracking**: Optional gradient norm logging for analysis

## Prerequisites

- Python 3.9.6 (recommended - tested with socket communication in tmux environments)
- uv package manager (install from [astral.sh/uv](https://astral.sh/uv))

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

The project uses two configuration files located in `src/smolcluster/configs/`:

### cluster_config_ddp.yaml
```yaml
host_ip: "10.10.0.1"        # IP address to bind the server (interface)
port: 65432                  # TCP port for communication
num_workers: 2               # Number of worker nodes to expect
timeout: 0.1                 # Timeout for gradient collection (seconds)
```

**Configuration Details:**
- `host_ip`: IP address to bind the server (interface). Set this to the server's network IP (e.g., WiFi network IP like 10.10.0.1).
- `port`: TCP port for socket communication. Ensure this port is open and not blocked by firewalls.
- `num_workers`: Number of worker processes/nodes. Must match the number of workers you plan to launch.
- `timeout`: How long the server waits for gradients from all workers before proceeding. Lower values = faster training but may skip slow workers.

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

