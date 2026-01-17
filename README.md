# SmolCluster

A distributed training framework for MNIST classification using PyTorch and socket-based communication. This project implements two distributed training algorithms: **Elastic Distributed Parallelism (EDP)** for asynchronous training and **Synchronous Parameter Server (SyncPS)** for synchronous training.

## Features

- **Multiple Training Algorithms**:
  - **Elastic Distributed Parallelism (EDP)**: Asynchronous parameter server with stale gradient handling
  - **Synchronous Parameter Server (SyncPS)**: Fully synchronous training with barrier-based coordination
- **Heterogeneous Cluster Support**: Train across Mac minis, Raspberry Pis, MacBooks, and Windows machines
- **Hybrid Network Topology**: Thunderbolt fabric for inter-Mac communication + Ethernet/WiFi for edge nodes
- **Socket Communication**: TCP socket-based communication with retry logic and connection resilience
- **Centralized Logging**: Grafana + Loki stack for real-time distributed log aggregation and visualization
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

## Cluster Topology

<img src="images/architecture.png" alt="Cluster Topology" width="100%">

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
- **Grafana Dashboard**: Visit [http://localhost:3000](http://localhost:3000) to view centralized logs (admin/admin)
- **W&B Dashboard**: Visit [wandb.ai](https://wandb.ai) to see detailed metrics:
  - Training/validation loss
  - Validation accuracy
  - Per-layer gradient norms (if `track_gradients: true`)
  - Step timing and throughput
- **Project Name**: `smolcluster`
- **Run Names**: Automatically generated with format: `{Algorithm}-{role}-{hostname}_rank{X}_lr{Y}_bs{Z}`

## Centralized Logging with Grafana + Loki

SmolCluster includes a distributed logging system that aggregates logs from all training nodes in real-time.

### Architecture

- **Grafana** (http://localhost:3000): Web UI for log visualization
- **Loki**: Log aggregation backend (receives logs from all nodes)
- **Promtail**: Log shipper (runs on each training node, sends logs to Loki)

### Quick Setup

1. **Install Docker** on your controller machine (the machine running the launch script):
   ```bash
   # macOS
   brew install --cask docker
   
   # Start Docker Desktop and ensure it's running
   ```

2. **Install Promtail** on all training nodes (minis, workers):
   ```bash
   # macOS (via Homebrew)
   brew install promtail
   
   # Linux
   curl -O -L "https://github.com/grafana/loki/releases/download/v2.9.0/promtail-linux-amd64.zip"
   unzip promtail-linux-amd64.zip
   chmod +x promtail-linux-amd64
   sudo mv promtail-linux-amd64 /usr/local/bin/promtail
   ```

3. **Launch training** (logging infrastructure starts automatically):
   ```bash
   bash launch_edp_gpt.sh
   ```
   
   The launch script automatically:
   - Starts Grafana + Loki in Docker containers on your controller
   - Detects and starts Promtail on each remote training node
   - Configures timezone handling (IST → UTC conversion)

4. **Access Grafana**:
   - Open http://localhost:3000
   - Login: `admin` / `admin`
   - Navigate to **Explore** → Select **Loki** datasource
   - Query examples:
     ```
     {job="smolcluster-worker"}              # All worker logs
     {job="smolcluster-server"}              # Server logs
     {host="worker-rank0-mini2"}             # Specific worker
     {level="ERROR"}                         # Only errors
     {job="smolcluster-worker"} |= "loss"    # Filter for "loss"
     ```

### Log Structure

Logs are automatically structured with labels for easy filtering:
- `job`: `smolcluster-server` or `smolcluster-worker`
- `component`: `server` or `worker`
- `host`: Extracted from filename (e.g., `worker-rank0-mini2`)
- `rank`: Worker rank (e.g., `0`, `1`)
- `level`: Log level (`INFO`, `ERROR`, `WARNING`)

### Troubleshooting Logging

If logs don't appear in Grafana:

1. **Check Docker containers are running**:
   ```bash
   docker ps | grep -E "loki|grafana"
   ```

2. **Verify Promtail is running on training nodes**:
   ```bash
   ssh mini2 "ps aux | grep promtail"
   ```

3. **Check Promtail logs for errors**:
   ```bash
   ssh mini2 "cat /tmp/promtail.log"
   ```

4. **Restart logging infrastructure**:
   ```bash
   cd logging
   docker-compose restart loki
   ```

### Manual Setup (Advanced)

If you need to manually manage the logging stack:

```bash
# Start Loki + Grafana on controller
cd logging
docker-compose up -d

# Start Promtail on server node
ssh mini1 "promtail -config.file=~/Desktop/smolcluster/logging/promtail-server-remote.yaml &"

# Start Promtail on worker nodes
ssh mini2 "promtail -config.file=~/Desktop/smolcluster/logging/promtail-worker-remote.yaml &"
ssh mini3 "promtail -config.file=~/Desktop/smolcluster/logging/promtail-worker-remote.yaml &"
```

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
│   │   └── gpt.py                    # GPT model for language modeling
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── common_utils.py           # Gradient ops, messaging, weight management
│   │   ├── data.py                   # Data partitioning utilities
│   │   ├── device.py                 # Device detection (MPS/CUDA/CPU)
│   │   ├── logging_utils.py          # Centralized logging setup
│   │   └── quantization.py           # Gradient quantization for EDP
│   ├── data/
│   │   ├── MNIST/                    # MNIST dataset (auto-downloaded)
│   │   │   └── raw/                  # Raw MNIST files
│   │   └── wikitext.py               # Wikitext-2 dataset utilities
│   ├── configs/
│   │   ├── cluster_config_edp.yaml   # EDP cluster network settings
│   │   ├── cluster_config_syncps.yaml # SyncPS cluster network settings
│   │   ├── nn_config.yaml            # Model/training settings
│   │   └── gpt_config.yaml           # GPT training configuration
│   └── docs/
│       └── setup_cluster.md          # Mac mini cluster setup guide
├── logging/
│   ├── docker-compose.yml            # Grafana + Loki containers
│   ├── loki-config.yaml              # Loki configuration
│   ├── promtail-server-remote.yaml   # Promtail config for server
│   └── promtail-worker-remote.yaml   # Promtail config for workers
├── launch_edp.sh                     # Launch EDP training
├── launch_edp_gpt.sh                 # Launch EDP with GPT training
├── launch_syncps.sh                  # Launch SyncPS training
├── pyproject.toml                    # Project dependencies and config
├── README.md                         # This file
└── .gitignore
```

## GPT Language Model Training

The project includes a complete GPT language model training setup for reproducing research papers:

### GPT Configuration (`gpt_config.yaml`)

All hyperparameters are centralized in `src/smolcluster/configs/gpt_config.yaml`:

```yaml
# Model Architecture
model:
  model_dim: 256          # Model dimension
  num_layers: 6           # Number of transformer layers
  num_heads: 4            # Attention heads
  ff_dim: 1024            # Feed-forward dimension
  dropout: 0.1            # Dropout rate
  max_seq_len: 128        # Maximum sequence length

# Training Hyperparameters
training:
  batch_size: null        # Auto: 32 (L=128) or 16 (L=256)
  epochs: null            # Auto: 30 (wikitext) or 20 (SNLI)
  learning_rate: 6e-4     # Learning rate
  weight_decay: 0.01      # AdamW weight decay
  grad_clip_norm: 1.0     # Gradient clipping

# Learning Rate Schedule
lr_schedule:
  warmup_iters: 100       # Warmup iterations
  min_lr: 6e-5           # Minimum learning rate

# Data & Logging
data:
  tokenizer: "openai-community/gpt2"
logging:
  project_name: "smolcluster-gpt-wikitext2"
```

### Training Commands

```bash
# Train with default config
uv run python src/smolcluster/train.py

# Override specific parameters
uv run python src/smolcluster/train.py --override training.batch_size=16 training.epochs=5

# Use different config file
uv run python src/smolcluster/train.py --config path/to/custom_config.yaml
```

### Features

- **Config-based**: All hyperparameters in YAML config file
- **Auto-calculated**: Batch size and epochs auto-set based on sequence length
- **Override support**: Command-line overrides for quick experimentation
- **W&B logging**: Automatic experiment tracking
- **Checkpointing**: Regular model saves with best validation tracking

## How It Works

### Elastic Distributed Parallelism (EDP)
1. **Asynchronous Operation**: Workers operate independently without waiting for each other
2. **Stale Gradient Handling**: Server accepts gradients even if they're from slightly older model versions
3. **Push-Pull Pattern**:
   - Workers compute local gradients and push to server
   - Workers periodically pull latest model weights from server
4. **Gradient Accumulation**: Server averages gradients from multiple workers before updating model
5. **Optional Quantization**: 8-bit quantization reduces communication overhead

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
- *
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
### Connection Issues
- Verify all nodes can ping the server IP: `ping 10.10.0.1`
- Check firewall rules allow port 65432
- Ensure server is started before workers
- Review connection logs in W&B for detailed error messages

### Training Issues
- If gradients explode: Enable gradient clipping in `nn_config.yaml`
- If loss doesn't decrease: Reduce learning rate or increase batch size
- For SyncPS stragglers: Increase timeout in `cluster_config_syncps.yaml`
- For EDP staleness: Decrease `worker_update_interval`

### W&B Issues
- Set `WANDB_API_KEY` environment variable for automatic login
- Check W&B run names match expected format
- Verify network connectivity to wandb.ai

## Contributing
Pull requests welcome! Please ensure your code follows the existing style and includes appropriate logging.

## License
MIT