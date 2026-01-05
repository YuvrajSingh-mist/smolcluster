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

### cluster_config.yaml
```yaml
host_ip: "0.0.0.0"          # Server bind address (0.0.0.0 = all interfaces)
worker_connect_ip: "172.18.3.80"  # IP address workers use to connect to server
port: 65432                  # TCP port for communication
num_workers: 2               # Number of worker nodes to expect
timeout: 0.1                 # Timeout for gradient collection (seconds)
```

**Configuration Details:**
- `host_ip`: Set to "0.0.0.0" to bind to all network interfaces. The server will listen on all available IPs.
- `worker_connect_ip`: The actual IP address that workers should connect to. This should be the server's IP on the network (e.g., WiFi network IP like 172.18.x.x).
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

**Configuration Details:**
- `batch_size`: Number of samples per training batch. Larger batches may improve stability but use more memory.
- `learning_rate`: Step size for gradient descent optimization. Typical values: 0.001-0.01.
- `num_epochs`: Number of complete passes through the training data.
- `eval_steps`: How often to evaluate the model on test data and log metrics (every N training steps).
- `track_gradients`: Whether to log gradient norms to Weights & Biases for analysis.
- `model.type`: Model architecture class name (must match a file in `models/` directory).
- `model.input_dim`: Input dimension (784 for MNIST flattened 28x28 images).
- `model.hidden`: Number of neurons in the hidden layer.
- `model.out`: Output dimension (10 for MNIST digit classification).
- `dataset.name`: Dataset to use (currently only MNIST is supported).

## Automated Cluster Launch

The `launch.sh` script automates the process of starting distributed training across multiple Mac mini nodes via SSH. It handles SSH connectivity checks, remote environment setup, and tmux session management.

### Prerequisites for launch.sh

- SSH access configured between your Mac minis (passwordless SSH with key-based authentication)
- tmux installed on all nodes: `brew install tmux`
- uv package manager installed on all nodes: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Remote project directory at `~/Desktop/smolcluster` (or update `REMOTE_PROJECT_DIR` in the script)
- Python 3.9.6 virtual environment set up on all nodes

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

### Customizing launch.sh

- **Node Names**: Update the node list in the script (currently mini1, mini2, mini3)
- **Remote Path**: Change `REMOTE_PROJECT_DIR` if your project is in a different location
- **Python Version**: Modify the venv creation command if using a different Python version
- **Session Names**: Adjust tmux session names if needed

## Usage

### 1. Start the Server

Open a terminal and start the central server:

```bash
cd src/smolcluster/NoRingReduce
../../../.venv/bin/python server.py
```

The server will:
- Initialize the model
- Load MNIST data
- Wait for worker connections
- Coordinate training and gradient averaging

### 2. Start Workers

For each worker, open a new terminal and run:

```bash
cd src/smolcluster/NoRingReduce
../../../.venv/bin/python worker.py
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
│   ├── NoRingReduce/       # Distributed training implementation
│   │   ├── server.py       # Central server coordinating training
│   │   └── worker.py       # Worker node for distributed training
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
│       ├── cluster_config.yaml  # Cluster network settings
│       └── nn_config.yaml       # Model/training settings
│   └── docs/
│       └── setup_cluster.md     # Mac mini cluster setup guide
├── smolcluster.egg-info/  # Package metadata (auto-generated)
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
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

## Troubleshooting

### Common Issues and Solutions

**"No route to host" errors:**
- Ensure all nodes are on the same network (WiFi recommended over Thunderbolt bridge)
- Check ARP cache: Run `arp -a` and `ping <worker_ip>` to warm up ARP table
- Use `worker_connect_ip` in cluster_config.yaml that matches your network (e.g., 172.18.x.x for WiFi)

**Socket connection timeouts:**
- Python 3.9.6 is recommended over 3.13 for socket reliability in tmux environments
- Increase `timeout` value in cluster_config.yaml if workers are slow to respond
- Check firewall settings and ensure the configured port (65432) is open

**SSH connectivity issues:**
- Set up passwordless SSH: `ssh-keygen -t rsa` then `ssh-copy-id mini1`, etc.
- Test SSH: `ssh -o ConnectTimeout=5 mini1 'echo "SSH OK"'`
- Ensure remote project path exists: `ssh mini1 'ls ~/Desktop/smolcluster'`

**tmux session issues:**
- Install tmux: `brew install tmux`
- Check sessions: `ssh mini1 'tmux ls'`
- Attach to session: `ssh mini1 'tmux attach -t server'`
- Kill stuck sessions: `ssh mini1 'tmux kill-session -t server'`

**Python environment issues:**
- Use Python 3.9.6 specifically: `uv venv --python 3.9.6 .venv`
- Activate venv: `source .venv/bin/activate`
- Install dependencies: `uv pip install -e .`
- Check Python version: `python --version`

**W&B logging issues:**
- Login to W&B: `wandb login`
- Check API key: `wandb status`
- Disable logging by setting `track_gradients: false` in nn_config.yaml

### Debug Commands

```bash
# Check network connectivity
ping 172.18.3.80  # Replace with your worker_connect_ip
arp -a           # Check ARP table

# Test socket connection
telnet 172.18.3.80 65432

# Check tmux sessions
ssh mini1 'tmux ls'
ssh mini2 'tmux ls'

# View recent logs
ssh mini1 'tail -50 ~/server.log'
ssh mini2 'tail -50 ~/worker1.log'

# Check Python environment
ssh mini1 'cd ~/Desktop/smolcluster && source .venv/bin/activate && python --version'
```

## Setting Up a Mac Mini Cluster
For setup instructions for creating a Mac mini cluster using Thunderbolt and SSH, refer to the [setup_cluster.md](src/smolcluster/docs/setup_cluster.md) guide.

