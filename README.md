# SmolCluster

A distributed training framework for MNIST classification using PyTorch and socket-based communication. This project implements two distributed training algorithms: **Elastic Distributed Parallelism (EDP)** for asynchronous training and **Synchronous Parameter Server (SyncPS)** for synchronous training.

## Features

- **Multiple Training Algorithms**:
  - **Elastic Distributed Parallelism (EDP)**: Asynchronous parameter server with stale gradient handling
  - **Synchronous Parameter Server (SyncPS)**: Fully synchronous training with barrier-based coordination
- **Heterogeneous Cluster Support**: Train across Mac minis, Raspberry Pis, MacBooks, and Windows machines
- **Hybrid Network Topology**: Thunderbolt fabric for inter-Mac communication + Ethernet/WiFi for edge nodes
- **Socket Communication**: TCP socket-based communication with retry logic and connection resilience
- **W&B Integration**: Automatic logging of training metrics with detailed run naming (hostname, LR, batch size)
- **Configurable**: YAML-based configuration for easy experimentation
- **MNIST Dataset**: Built-in support for MNIST handwritten digit recognition
- **Gradient Tracking**: Optional gradient norm logging and gradient clipping
- **Polyak Averaging**: Smooth weight updates using exponential moving average
- **Quantization Support**: Optional gradient quantization for bandwidth-limited links (EDP)

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

## Topology

[Cluster Topology Diagram](src/smolcluster/docs/cluster_topology.png)


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

The project uses configuration files located in `src/smolcluster/configs/`:

### cluster_config_edp.yaml (Elastic Distributed Parallelism)
```yaml
host_ip:
  mini1: "10.10.0.1"      # Server on Thunderbolt network
  macbook: "10.10.0.1"    # Worker (connects to server)
  pi5: "192.168.50.1"     # Worker via Ethernet
  win: "10.10.0.1"        # Worker (Windows machine)

port: 65432
num_workers: 3              # Number of worker nodes
workers: [pi5, win, macbook]  # Worker hostnames
server: mini1                 # Server hostname
worker_update_interval: 10    # Steps between weight pulls from server
use_quantization: false       # Enable gradient quantization
seed: 42
```

**EDP Configuration Details:**
- `host_ip`: Maps hostnames to their network addresses (all workers connect to server at 10.10.0.1)
- `port`: TCP port for socket communication (65432 by default)
- `num_workers`: Number of worker nodes expected
- `workers`: List of worker hostnames (must match keys in `host_ip`)
- `server`: Server hostname (must match key in `host_ip`)
- `worker_update_interval`: How often workers pull latest weights from server (in training steps)
- `use_quantization`: Enable 8-bit quantization for gradient communication (reduces bandwidth)
- `seed`: Random seed for reproducible data partitioning

### cluster_config_syncps.yaml (Synchronous Parameter Server)
```yaml
host_ip:
  mini1: "10.10.0.1"      # Server on Thunderbolt network
  macbook: "10.10.0.1"    # Worker (connects to server)
  pi5: "192.168.50.1"     # Worker via Ethernet
  win: "10.10.0.1"        # Worker (Windows machine)

port: 65432
num_workers: 3              # Number of worker nodes
workers: [pi5, win, macbook]  # Worker hostnames
server: mini1                 # Server hostname
worker_update_interval: 4     # Steps between full weight synchronization
timeout: 0.1                  # Server timeout for gradient collection (seconds)
seed: 42
```

**SyncPS Configuration Details:**
- Similar to EDP config but with synchronous semantics
- `worker_update_interval`: How often workers pull and apply Polyak-averaged weights (in steps)
- `timeout`: How long server waits for gradients before proceeding (for handling stragglers)
- No quantization support (designed for low-latency networks)

### nn_config.yaml
```yaml
batch_size: 32
learning_rate: 0.001
num_epochs: 2
eval_steps: 200             # Evaluate every N steps
track_gradients: true       # Log gradient norms to W&B
polyak_alpha: 0.5          # Polyak averaging coefficient (0.5 = equal blend)
gradient_clipping:
  enabled: true            # Enable gradient clipping
  max_norm: 1.0            # Maximum gradient norm

model:
  type: SimpleNN
  input_dim: 784
  hidden: 128
  out: 10

dataset:
  name: MNIST
```

**Training Configuration Details:**
- `batch_size`: Batch size per worker
- `learning_rate`: Learning rate for gradient descent
- `num_epochs`: Total training epochs
- `eval_steps`: How often to evaluate on validation set
- `track_gradients`: Log per-layer gradient norms to W&B
- `polyak_alpha`: Blending factor for Polyak averaging (higher = more server influence)
- `gradient_clipping`: Prevent exploding gradients by clipping to max norm

## Automated Cluster Launch

Two launch scripts are provided for the different training algorithms:

### `launch_edp.sh` (Elastic Distributed Parallelism)
Launches the asynchronous parameter server training:

```bash
bash launch_edp.sh
```

### `launch_syncps.sh` (Synchronous Parameter Server)
Launches the synchronous parameter server training:

```bash
bash launch_syncps.sh
```

### How the Launch Scripts Work

1. **tmux Session Management**: Each process runs in a separate tmux pane for easy monitoring
2. **Server Launch**: Starts the central parameter server in the first pane
3. **Worker Launch**: Starts worker processes in subsequent panes, each connecting to the server
4. **Automatic Configuration**: Workers automatically receive their rank and hostname from script arguments

### Monitoring Training

After launching, you can monitor the training in the tmux window:
- Each pane shows real-time logs from server/workers
- Use `Ctrl+B` then arrow keys to navigate between panes
- Use `Ctrl+B` then `[` to scroll through pane history
- Detach with `Ctrl+B` then `d`, reattach with `tmux attach`

## Usage

### Quick Start (Recommended)

Use the automated launch scripts:

```bash
# For Elastic Distributed Parallelism (asynchronous)
bash launch_edp.sh

# For Synchronous Parameter Server
bash launch_syncps.sh
```

### Manual Launch (For Debugging)

#### EDP (Elastic Distributed Parallelism)

**1. Start the Server:**
```bash
cd src/smolcluster/algorithms/EDP
uv run server.py
```

**2. Start Workers:**
```bash
cd src/smolcluster/algorithms/EDP
uv run worker.py <worker_id> <hostname>
# Example: uv run worker.py 1 macbook
```

#### SyncPS (Synchronous Parameter Server)

**1. Start the Server:**
```bash
cd src/smolcluster/algorithms/SynchronousPS
uv run server.py
```

**2. Start Workers:**
```bash
cd src/smolcluster/algorithms/SynchronousPS
uv run worker.py <worker_id> <hostname>
# Example: uv run worker.py 1 pi5
```

### Monitor Training

- **Console Logs**: View real-time training progress in each tmux pane
- **W&B Dashboard**: Visit [wandb.ai](https://wandb.ai) to see detailed metrics:
  - Training/validation loss
  - Validation accuracy
  - Per-layer gradient norms (if `track_gradients: true`)
  - Step timing and throughput
- **Project Name**: `smolcluster`
- **Run Names**: Automatically generated with format: `{Algorithm}-{role}-{hostname}_rank{X}_lr{Y}_bs{Z}`

## Project Structure

```
smolcluster/
├── src/smolcluster/
│   ├── algorithms/
│   │   ├── EDP/                      # Elastic Distributed Parallelism
│   │   │   ├── server.py             # Asynchronous parameter server
│   │   │   └── worker.py             # Asynchronous worker
│   │   └── SynchronousPS/            # Synchronous Parameter Server
│   │       ├── server.py             # Synchronous parameter server
│   │       └── worker.py             # Synchronous worker
│   ├── models/
│   │   ├── __init__.py
│   │   └── SimpleNN.py               # Simple neural network model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── common_utils.py           # Gradient ops, messaging, weight management
│   │   ├── data.py                   # Data partitioning utilities
│   │   ├── device.py                 # Device detection (MPS/CUDA/CPU)
│   │   └── quantization.py           # Gradient quantization for EDP
│   ├── data/
│   │   └── MNIST/                    # MNIST dataset (auto-downloaded)
│   │       └── raw/                  # Raw MNIST files
│   ├── configs/
│   │   ├── cluster_config_edp.yaml   # EDP cluster network settings
│   │   ├── cluster_config_syncps.yaml # SyncPS cluster network settings
│   │   └── nn_config.yaml            # Model/training settings
│   └── docs/
│       └── setup_cluster.md          # Mac mini cluster setup guide
├── launch_edp.sh                     # Launch EDP training
├── launch_syncps.sh                  # Launch SyncPS training
├── pyproject.toml                    # Project dependencies and config
├── README.md                         # This file
└── .gitignore
```

## How It Works
### Elastic Distributed Parallelism (EDP)
1. **Asynchronous Operation**: Workers operate independently without waiting for each other
2. **Stale Gradient Handling**: Server accepts gradients even if they're from slightly older model versions
3. **Optional Quantization**: 8-bit quantization reduces communication overhead

### Synchronous Parameter Server (SyncPS)
1. **Barrier-Based Synchronization**: All workers must complete each step before proceeding
2. **Fresh Gradients Only**: Server only accepts gradients from the current training step
3. **Synchronous Gradient Reduce**:
   - Workers send gradients for step N
   - Server waits for all workers (with timeout)
   - Server averages gradients and updates model
4. **Periodic Weight Sync**: Workers pull and apply Polyak-averaged weights every K steps
5. **Polyak Averaging**: Smooth weight updates using exponential moving average

### Common Elements
- **Data Partitioning**: MNIST training data is split across workers using deterministic seed-based approach
- **Heterogeneous Support**: Each worker can have different compute capabilities
- **Fault Tolerance**: Connection retry logic handles temporary network issues
- **W&B Logging**: Comprehensive metrics tracking across all nodes

## **Push-Pull Pattern**:
   - Workers compute local gradients and push to server
   - Workers periodically pull latest model weights from server
4. **Gradient Accumulation**: Server averages gradients from multiple workers before updating model
5. Algorithm Comparison

| Feature | EDP | SyncPS |
|---------|-----|--------|
| **Synchronization** | Asynchronous | Synchronous |
| **Gradient Staleness** | Tolerates stale gradients | Requires fresh gradients |
| **Barrier Points** | None | Every training step |
| **Throughput** | Higher (no waiting) | Lower (workers wait) |
| **Convergence** | May be slower | Generally faster |
| **Fault Tolerance** | Better (workers independent) | Weaker (timeout required) |
| **Network Efficiency** | Quantization supported | Raw gradients only |
| **Best For** | Heterogeneous clusters, high-latency networks | Homogeneous clusters, low-latency networks |

## Setting Up a Cluster
For setup instructions for creating a Mac mini cluster using Thunderbolt and SSH, refer to the [setup_cluster.md](src/smolcluster/docs/setup_cluster.md) guide.

## Troubleshooting

4. **Periodic Weight Sync**: Workers pull and apply Polyak-averaged weights every K steps
5. **Polyak Averaging**: Smooth weight updates using exponential moving average

### Common Elements
- **Data Partitioning**: MNIST training data is split across workers using deterministic seed-based approach
- **Heterogeneous Support**: Each worker can have different compute capabilities
- **Fault Tolerance**: Connection retry logic handles temporary network issues
- **W&B Logging**: Comprehensive metrics tracking across all node
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

