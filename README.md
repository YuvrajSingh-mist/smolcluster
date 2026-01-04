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

- Python 3.13 or higher
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

### cluster_config.yaml
```yaml
host_ip: null  # null = auto-detect local IP
port: 65432
num_workers: 2  # Number of worker nodes
timeout: 0.1   # Timeout for gradient collection (seconds)
```

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

## Usage

### 1. Start the Server

Open a terminal and start the central server:

```bash
cd src/smolcluster
uv run ./server.py
```

The server will:
- Initialize the model
- Load MNIST data
- Wait for worker connections
- Coordinate training and gradient averaging

### 2. Start Workers

For each worker, open a new terminal and run:

```bash
cd src/smolcluster
uv run ./worker.py
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
│   ├── server.py          # Central server coordinating training
│   ├── worker.py          # Worker node for distributed training
│   ├── main.py           # (Optional) Standalone training script
│   ├── models/
│   │   ├── __init__.py
│   │   └── SimpleNN.py   # Simple neural network model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── common_utils.py  # Gradient ops, messaging
│   │   ├── data.py         # Data partitioning utilities
│   │   └── device.py       # Device detection
│   └── configs/
│       ├── cluster_config.yaml  # Cluster settings
│       └── nn_config.yaml       # Model/training settings
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

For setup instructions for creating a Mac mini cluster using Thunderbolt and SSH, refer to the [setup_cluster.md](src/smolcluster/docs/setup_cluster.md) guide.

