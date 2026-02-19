# Smolcluster

**Website:** [smolcluster.com](https://smolcluster.com)

A distributed deep learning library for training neural networks across heterogeneous hardware using PyTorch and socket-based communication.

## Features

- **Distributed Training Algorithms**: FSDP (ZeRO-optimized), Classic Data Parallelism (All-Reduce), Elastic Distributed Parallelism (EDP), Synchronous Parameter Server (SyncPS), and Model Parallelism
- **Heterogeneous Hardware**: Mac minis, Raspberry Pis, MacBooks, and Windows machines
- **Model Support**: MNIST, GPT-2, and custom neural networks
- **Distributed Inference**: Model parallelism with streaming token generation
- **Centralized Logging**: Grafana + Loki for real-time log aggregation
- **Web Interface**: React-based chat UI for GPT inference
- **Experiment Tracking**: W&B integration with automatic metrics logging

## Quick Start

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/YuvrajSingh-mist/smolcluster.git
cd smolcluster
uv sync

# Launch training (example)
bash scripts/launch_edp_train_gpt.sh
```

## Cluster Topology

<img src="images/architecture.png" alt="Cluster Topology" width="100%">

## Documentation

- **[Configuration Guide](docs/configuration.md)** - Cluster and model configuration
- **[Training Guide](docs/training.md)** - Training algorithms and usage
- **[Logging Setup](docs/logging.md)** - Grafana + Loki distributed logging
- **[Cluster Setup](docs/setup_cluster.md)** - Hardware setup and networking
- **[Inference Guide](docs/inference.md)** - Model parallelism inference

## Training Algorithms

### FSDP (Fully Sharded Data Parallel)
ZeRO-optimized data parallelism with configurable optimizer state partitioning. Best for memory-constrained setups and large models.

```bash
bash scripts/launch_fsdp_train_gpt.sh
```

**Features:**
- ZeRO Stage 0: All-Reduce (classic data parallelism)
- ZeRO Stage 1: Optimizer state partitioning (~1/N memory per worker)
- Bandwidth-optimized weight broadcasting (only owned parameters)
- Configurable bounded staleness (0 = strict sync, K > 0 = async up to K steps)
- Real-time staleness monitoring via WandB

### Classic Data Parallelism (ClassicDP)
All-Reduce based data parallelism with bounded staleness. Best for balanced clusters with moderate network latency.

```bash
bash scripts/launch_dp_train_gpt.sh
```

**Features:**
- All-to-all gradient averaging (ring all-reduce)
- Configurable bounded staleness (0 = strict sync, K > 0 = async up to K steps)
- Real-time staleness monitoring via WandB
- Automatic stale gradient cleanup

### Elastic Distributed Parallelism (EDP)
Asynchronous data parallelism with stale gradient tolerance. Best for heterogeneous clusters.

```bash
bash scripts/launch_edp_train_gpt.sh
```

### Synchronous Parameter Server (SyncPS)
Synchronous data parallelism with barrier coordination. Best for homogeneous clusters.

```bash
bash scripts/launch_syncps_train_gpt.sh
```

### Model Parallelism (MP)
Layer-wise model distribution. Best for large models and inference serving.

```bash
bash scripts/inference/launch_mp_inference.sh
bash scripts/inference/launch_api.sh
```

See [training.md](docs/training.md) for detailed algorithm comparison and usage.

## Monitoring

### Weights & Biases
Real-time experiment tracking at [wandb.ai](https://wandb.ai)
- Training/validation metrics
- Per-layer gradient norms
- Hardware utilization

### Grafana + Loki
Centralized log aggregation at [http://localhost:3000](http://localhost:3000)
- Distributed logs from all nodes
- Real-time log queries
- Error tracking

See [logging.md](docs/logging.md) for setup instructions.

## Project Structure

```
smolcluster/
├── docs/                           # Documentation
│   ├── configuration.md            # Config guide
│   ├── training.md                 # Training guide
│   ├── logging.md                  # Logging setup
│   ├── inference.md                # Inference guide
│   └── setup_cluster.md            # Hardware setup
├── src/smolcluster/
│   ├── algorithms/
│   │   ├── EDP/                    # Elastic Distributed Parallelism
│   │   ├── DataParallelism/        # Data Parallelism implementations
│   │   │   ├── ClassicDP/          # Classic All-Reduce Data Parallelism
│   │   │   └── SynchronousPS/      # Synchronous Parameter Server
│   │   ├── FSDP/                   # Fully Sharded Data Parallelism
│   │   ├── ModelParallelism/       # Model Parallelism
│   │   └── ModelParallelismPipeline/  # Pipeline Model Parallelism
│   ├── models/                     # Neural network models
│   ├── utils/                      # Utilities and helpers
│   ├── data/                       # Datasets
│   ├── configs/                    # YAML configurations
│   └── chat/                       # Web inference interface
├── scripts/                        # Launch scripts
├── logging/                        # Grafana + Loki setup
└── pyproject.toml                  # Dependencies
```

## Contributing
Pull requests welcome! Please ensure your code follows the existing style and includes appropriate logging.

## License
MIT