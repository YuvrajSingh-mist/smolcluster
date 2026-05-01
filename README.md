# Smolcluster

**Website:** [smolcluster.com](https://smolcluster.com)

A distributed deep learning library for training neural networks across heterogeneous hardware using PyTorch and socket-based communication.

## grove TUI — zero setup, runs immediately

- The fastest way to see smolcluster's training algorithms in action. **grove** is a terminal dashboard that allows you to discover and connect to both mac-based and Linux-based nodes seamlessly (NO SSH REQUIRED!), giving you a live unified view across every node in your cluster — no config files, no IPs, no extra services.

- Workers auto-discover the coordinator via AirDrop/mDNS on Mac or TCP/mDNS on Jetson and Linux. Every training algorithm (FSDP, ClassicDP, EDP, SyncPS, EP, GRPO) reports into it automatically — no code changes needed.

> For full setup instructions see the [quickstart guide](https://www.smolcluster.com/quickstart.html). For grove internals and transport options see the [grove README](grove/README.md).

## Features

- **Distributed Training Algorithms**: Fully Sharded Data Parallel (ZeRO-optimized), Classic Data Parallelism (All-Reduce), Elastic Distributed Parallelism, Synchronous Parameter Server (SyncPS), Expert Parallelism (EP), and Model Parallelism
- **Heterogeneous Hardware**: Mac minis, Raspberry Pis, MacBooks, and Windows machines
- **Model Support**: MNIST, GPT-2, and custom neural networks
- **Distributed Inference**: Model parallelism with streaming token generation
- **Web Interface**: React-based chat UI for GPT inference
- **Experiment Tracking**: W&B integration with automatic metrics logging

## Quick Start

Refer to the [quickstart](https://www.smolcluster.com/quickstart.html) guide for an quick and easy setup or [MANUAL.md](docs/manual.md) for a step-by-step manual setup guide and to run your first training/inference algorithm on smolcluster.



## Cluster Topology

<img src="images/architecture.png" alt="Cluster Topology" width="100%">

## Documentation

- **[Cluster Setup Guide](docs/setup_cluster.md)** - Complete setup for distributed training cluster
- **[Network Configuration Guide](docs/networking.md)** - Detailed networking setup (Thunderbolt + Ethernet)
- **[Training Guide](docs/training.md)** - Training algorithms and usage
- **[Configuration Guide](docs/configuration.md)** - Cluster and model configuration
- **[Inference Guide](docs/inference.md)** - Model parallelism inference
- **[Inference API Reference](docs/api.md)** - HTTP + SSE endpoints for MP and DP
- **[Logging Setup](docs/logging.md)** - Distributed log monitoring via the dashboard

## Training Algorithms

### Fully Sharded Data Parallel
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

### Elastic Distributed Parallelism
Asynchronous data parallelism with stale gradient tolerance. Best for heterogeneous clusters.

```bash
bash scripts/launch_edp_train_gpt.sh
```

### Synchronous Parameter Server (SyncPS)
Synchronous data parallelism with barrier coordination. Best for homogeneous clusters.

```bash
bash scripts/launch_syncps_train_gpt.sh
```

### Expert Parallelism (EP)
Mixture-of-Experts training with experts sharded across nodes. Best for scaling MoE models efficiently across heterogeneous hardware.

```bash
bash scripts/training/launch_ep_train_moe.sh
```

### Model Parallelism (MP)
Layer-wise model distribution. Best for large models and inference serving.

```bash
bash scripts/inference/launch_inference.sh --algorithm mp
bash scripts/inference/launch_api.sh
```

See [training.md](docs/training.md) for detailed algorithm comparison and usage.

## Monitoring

### Weights & Biases
Real-time experiment tracking at [wandb.ai](https://wandb.ai)
- Training/validation metrics
- Per-layer gradient norms
- Hardware utilization

See [logging.md](docs/logging.md) for log monitoring setup.

## Project Structure

```
smolcluster/
├── docs/                           # Documentation
│   ├── configuration.md            # Config guide
│   ├── training.md                 # Training guide
│   ├── logging.md                  # Logging setup
│   ├── inference.md                # Inference guide
│   ├── api.md                      # Inference API reference
│   └── setup_cluster.md            # Hardware setup
├── src/smolcluster/
│   ├── algorithms/
│   │   ├── Elastic Distributed Parallelism/                    # Elastic Distributed Parallelism
│   │   ├── DataParallelism/        # Data Parallelism implementations
│   │   │   ├── ClassicDP/          # Classic All-Reduce Data Parallelism
│   │   │   └── SynchronousPS/      # Synchronous Parameter Server
│   │   ├── ExpertParallelism/      # Expert Parallelism (MoE)
│   │   ├── Fully Sharded Data Parallel/                   # Fully Sharded Data Parallelism
│   │   ├── ModelParallelism/       # Model Parallelism
│   │   └── ModelParallelismPipeline/  # Pipeline Model Parallelism
│   ├── models/                     # Neural network models
│   ├── utils/                      # Utilities and helpers
│   ├── data/                       # Datasets
│   ├── configs/                    # YAML configurations
│   └── chat/                       # Web inference interface
├── scripts/                        # Launch scripts
├── logging/                        # Cluster log files
└── pyproject.toml                  # Dependencies
```

## Contributing
Pull requests welcome! Please ensure your code follows the existing style and includes appropriate logging.

## License
MIT