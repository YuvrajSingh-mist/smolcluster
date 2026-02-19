# Configuration Guide

Smolcluster uses YAML configuration files to manage cluster topology, model architecture, and training parameters. All configuration files are located in `src/smolcluster/configs/`.

---

## Table of Contents

- [Cluster Configuration](#cluster-configuration)
  - [FSDP (Fully Sharded Data Parallel)](#cluster_config_fsdpyaml-fully-sharded-data-parallel)
  - [Classic Data Parallelism (ClassicDP)](#cluster_config_classicdpyaml-classic-data-parallelism)
  - [Elastic Distributed Parallelism (EDP)](#cluster_config_edpyaml-elastic-distributed-parallelism)
  - [Synchronous Parameter Server (SyncPS)](#cluster_config_syncpsyaml-synchronous-parameter-server)
  - [Model Parallelism (MP)](#cluster_config_mpyaml-model-parallelism)
- [Model Configuration](#model-configuration)
  - [Simple Neural Network](#nn_configyaml-simple-neural-network)
  - [GPT Language Model](#gpt_configyaml-gpt-language-model)
- [Inference Configuration](#inference-configuration)
  - [Model Configuration](#model_config_inferenceyaml)
  - [Cluster Configuration](#cluster_config_inferenceyaml)
- [Network Topology](#network-topology)
- [Distributed Inference Deployment](#distributed-inference-deployment)
- [Command-Line Overrides](#command-line-overrides)

---

## Cluster Configuration

### cluster_config_fsdp.yaml (Fully Sharded Data Parallel)

ZeRO-optimized data parallelism with configurable optimizer state partitioning, ideal for memory-constrained setups and large models.

```yaml
# FSDP stage selection
fsdp_stage: 1                  # 0: All-Reduce, 1: ZeRO Stage 1 optimizer partitioning

# Bounded staleness for gradient synchronization
staleness_bound: 5             # Allow workers to be 5 steps apart (0 = strict sync)

# Network buffer sizes (in MB) - device-specific optimizations
buffer_size:
  mini1: 8                     # Mac Mini - 8MB (Thunderbolt 40Gbps)
  mini2: 8                     # Mac Mini - 8MB
  mini3: 8                     # Mac Mini - 8MB
  macbook: 8                   # MacBook - 8MB
  pi4: 2                       # Raspberry Pi 4 - 2MB (Gigabit Ethernet)
  pi5: 2                       # Raspberry Pi 5 - 2MB

# Network metrics tracking
track_network_metrics: true    # Enable bandwidth, latency, buffer usage tracking
metrics_log_interval: 50       # Log network metrics every N steps

# Cluster configuration
num_workers: 3                 # Number of worker nodes

# All-to-all topology (fully connected)
allToAllTopology:
  workers:
    regular:
      - hostname: mini1
        rank: 0
        port: 65432
        ip: \"10.10.0.1\"
      - hostname: mini2
        rank: 1
        port: 65433
        ip: \"10.10.0.2\"
      - hostname: mini3
        rank: 2
        port: 65434
        ip: \"10.10.0.3\"

# Model parallelism specific configs
num_nodes: 3                   # Number of nodes in cluster
num_layers: 6                  # Number of transformer blocks
model_name: causal_gpt2        # Model architecture identifier
seed: 42                       # Random seed
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `fsdp_stage` | int | ZeRO optimization stage: 0 (All-Reduce), 1 (Optimizer Partitioning) |
| `staleness_bound` | int | Maximum step difference (0 = strict sync, K = bounded async) |
| `buffer_size` | dict | Network buffer sizes per device (MB) for optimized throughput |
| `track_network_metrics` | bool | Enable bandwidth/latency tracking |
| `metrics_log_interval` | int | Steps between metric logging |
| `num_workers` | int | Total number of workers in cluster |
| `allToAllTopology` | dict | Fully connected worker topology (rank, port, IP per worker) |
| `num_nodes` | int | Number of nodes (same as num_workers) |
| `num_layers` | int | Model layers for partitioning (used in ZeRO Stage 1) |
| `model_name` | str | Model architecture identifier |
| `seed` | int | Random seed for reproducibility |

**FSDP Stage Details:**
- **Stage 0 (All-Reduce)**: Classic data parallelism with full model replicas on each worker. All gradients averaged, all parameters synchronized.
- **Stage 1 (ZeRO Optimizer Partitioning)**: Each worker owns subset of model layers and only updates those parameters. Memory savings: ~1/N optimizer states per worker. Bandwidth optimization: only owned parameters broadcasted.

### cluster_config_classicdp.yaml (Classic Data Parallelism)

All-Reduce based data parallelism with bounded staleness, ideal for balanced clusters with moderate network latency.

```yaml
# Bounded staleness for gradient synchronization
staleness_bound: 5             # Allow workers to be 5 steps apart (0 = strict sync)

# Network buffer sizes and metrics (same as FSDP)
buffer_size:
  mini1: 8
  mini2: 8
  mini3: 8

track_network_metrics: true
metrics_log_interval: 50

# Cluster configuration (all-to-all topology)
num_workers: 3
allToAllTopology:
  workers:
    regular:
      - hostname: mini1
        rank: 0
        port: 65432
        ip: \"10.10.0.1\"
      - hostname: mini2
        rank: 1
        port: 65433
        ip: \"10.10.0.2\"
      - hostname: mini3
        rank: 2
        port: 65434
        ip: \"10.10.0.3\"

seed: 42
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `staleness_bound` | int | Maximum step difference (0 = strict sync, K = bounded async) |
| `buffer_size` | dict | Network buffer sizes per device (MB) |
| `track_network_metrics` | bool | Enable bandwidth/latency tracking |
| `num_workers` | int | Total number of workers |
| `allToAllTopology` | dict | Fully connected worker topology |
| `seed` | int | Random seed for reproducibility |

### cluster_config_edp.yaml (Elastic Distributed Parallelism)

Asynchronous data parallelism with stale gradient tolerance, ideal for heterogeneous clusters.

```yaml
host_ip:
  mini1: "10.10.0.1"           # Server on Thunderbolt network
  macbook: "10.10.0.1"         # Worker (connects to server)
  pi5: "192.168.50.1"          # Worker via Ethernet
  win: "10.10.0.1"             # Worker (Windows machine)

port: 65432                    # TCP communication port
num_workers: 3                 # Number of worker nodes
workers: [pi5, win, macbook]   # Worker hostnames
server: mini1                  # Server hostname
worker_update_interval: 10     # Steps between weight pulls
use_quantization: false        # Enable gradient quantization
seed: 42                       # Random seed
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `host_ip` | dict | Maps hostnames to network IP addresses |
| `port` | int | TCP port for socket communication (default: 65432) |
| `num_workers` | int | Number of worker nodes in the cluster |
| `workers` | list | Worker hostnames (must match `host_ip` keys) |
| `server` | str | Server hostname (must match `host_ip` key) |
| `worker_update_interval` | int | Training steps between weight synchronization |
| `use_quantization` | bool | Enable 8-bit gradient quantization |
| `seed` | int | Random seed for reproducible data partitioning |

### cluster_config_syncps.yaml (Synchronous Parameter Server)

Synchronous data parallelism with barrier coordination for homogeneous clusters.

```yaml
host_ip:
  mini1: "10.10.0.1"
  macbook: "10.10.0.1"
  pi5: "192.168.50.1"
  win: "10.10.0.1"

port: 65432                    # TCP communication port
num_workers: 3                 # Number of worker nodes
workers: [pi5, win, macbook]   # Worker hostnames
server: mini1                  # Server hostname
worker_update_interval: 4      # Steps between weight synchronization
timeout: 0.1                   # Gradient collection timeout (seconds)
seed: 42                       # Random seed
```

#### Parameters

Inherits all parameters from EDP configuration, with the following differences:

| Parameter | Type | Description |
|-----------|------|-------------|
| `worker_update_interval` | int | Steps between Polyak-averaged weight updates |
| `timeout` | float | Server timeout for gradient collection (handles stragglers) |

### cluster_config_mp.yaml (Model Parallelism)

Layer-wise model distribution for training and inference of large models.

```yaml
host_ip:
  mini1: "10.10.0.1"           # Server node
  mini2: "10.10.0.2"           # Worker 1
  mini3: "10.10.0.3"           # Worker 2

port: 65432                    # TCP communication port
num_workers: 2                 # Number of workers (total nodes - 1)
workers: [mini2, mini3]        # Worker hostnames
server: mini1                  # Server hostname
seed: 42                       # Random seed
timeout: 300                   # Inference request timeout (seconds)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `host_ip` | dict | Maps hostnames to network IP addresses |
| `port` | int | TCP port for socket communication |
| `num_workers` | int | Number of workers (total nodes - 1) |
| `workers` | list | Worker hostnames for layer distribution |
| `server` | str | Server hostname (coordinates inference) |
| `timeout` | int | Timeout for inference requests (seconds) |
| `seed` | int | Random seed for reproducibility |

#### Use Cases

- **Training**: Layer-wise distribution for models exceeding single-device memory
- **Inference**: Distributed GPT-2 generation with streaming token output

---

## Model Configuration

### nn_config.yaml (Simple Neural Network)

Configuration for simple feedforward networks (e.g., MNIST classification).

```yaml
batch_size: 32                 # Batch size per worker
learning_rate: 0.001           # Learning rate
num_epochs: 2                  # Total training epochs
eval_steps: 200                # Evaluation frequency
track_gradients: true          # Log gradient norms to W&B
polyak_alpha: 0.5              # Polyak averaging coefficient

gradient_clipping:
  enabled: true
  max_norm: 1.0

model:
  type: SimpleNN
  input_dim: 784               # MNIST input dimension
  hidden: 128                  # Hidden layer size
  out: 10                      # Number of classes

dataset:
  name: MNIST
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_size` | int | Batch size per worker |
| `learning_rate` | float | Learning rate for gradient descent |
| `num_epochs` | int | Total training epochs |
| `eval_steps` | int | Steps between evaluations |
| `track_gradients` | bool | Enable per-layer gradient norm logging (W&B) |
| `polyak_alpha` | float | Polyak averaging coefficient (0.0-1.0) |
| `gradient_clipping.enabled` | bool | Enable gradient clipping |
| `gradient_clipping.max_norm` | float | Maximum gradient norm |

### gpt_config.yaml (GPT Language Model)

Configuration for GPT-style transformer models (e.g., WikiText-2, SNLI datasets).

```yaml
model:
  model_dim: 256               # Model embedding dimension
  num_layers: 6                # Transformer layers
  num_heads: 4                 # Multi-head attention heads
  ff_dim: 1024                 # Feed-forward dimension
  dropout: 0.1                 # Dropout rate
  max_seq_len: 128             # Maximum sequence length

training:
  batch_size: null             # Auto: 32 (L=128) or 16 (L=256)
  epochs: null                 # Auto: 30 (wikitext) or 20 (SNLI)
  learning_rate: 6e-4          # AdamW learning rate
  weight_decay: 0.01           # AdamW weight decay
  grad_clip_norm: 1.0          # Gradient clipping threshold

lr_schedule:
  warmup_iters: 100            # Learning rate warmup steps
  min_lr: 6e-5                 # Minimum learning rate

data:
  tokenizer: "openai-community/gpt2"

logging:
  project_name: "smolcluster-gpt-wikitext2"
```

#### Parameters

**Model Architecture:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_dim` | int | Model embedding dimension |
| `num_layers` | int | Number of transformer blocks |
| `num_heads` | int | Multi-head attention heads |
| `ff_dim` | int | Feed-forward network dimension |
| `dropout` | float | Dropout rate for regularization |
| `max_seq_len` | int | Maximum sequence length |

**Training Configuration:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_size` | int/null | Batch size (auto-calculated if null) |
| `epochs` | int/null | Training epochs (auto-calculated if null) |
| `learning_rate` | float | AdamW optimizer learning rate |
| `weight_decay` | float | AdamW weight decay coefficient |
| `grad_clip_norm` | float | Maximum gradient norm for clipping |

**Auto-Calculated Parameters:**
- `batch_size`: 32 (L=128) or 16 (L=256)
- `epochs`: 30 (WikiText-2) or 20 (SNLI)

---

## Inference Configuration

### model_config_inference.yaml

Model parameters and decoding strategies for distributed GPT-2 inference.

```yaml
causal_gpt2:
  hf_model_name: "openai-community/gpt2"  # HuggingFace model identifier
  weights_model_name: "gpt2"               # Key in model_weights.yaml
  num_nodes: 3                             # Nodes for model parallelism
  num_layers: 12                           # Total transformer layers
  max_new_tokens: 256                      # Maximum tokens to generate
  ctx_length: 1024                         # Context length
  
  active_decoding_strategy: "top_k"        # Active strategy
  
  decoding_strategies:
    greedy:
      temperature: 1.0
    sampling:
      temperature: 1.0
    top_p:
      temperature: 1.0
      p: 0.9
    top_k:
      temperature: 1.0
      k: 40
```

#### Parameters

**Model Configuration:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hf_model_name` | str | HuggingFace model identifier |
| `weights_model_name` | str | Key in `model_weights.yaml` for downloading |
| `num_nodes` | int | Number of nodes for layer distribution |
| `num_layers` | int | Total transformer layers (GPT-2: 12) |
| `max_new_tokens` | int | Maximum tokens to generate per request |
| `ctx_length` | int | Maximum context length |

**Decoding Strategies:**

| Strategy | Parameters | Description |
|----------|------------|-------------|
| `greedy` | `temperature` | Selects highest probability token |
| `sampling` | `temperature` | Samples from probability distribution |
| `top_p` | `temperature`, `p` | Nucleus sampling (top-p threshold) |
| `top_k` | `temperature`, `k` | Samples from top-k tokens |

- `temperature`: Sampling temperature (1.0 = neutral, >1.0 = random, <1.0 = conservative)
- `p`: Top-p threshold for nucleus sampling (0.0-1.0)
- `k`: Number of top tokens to consider

### cluster_config_inference.yaml

```yaml
host_ip:
  mini1: "10.10.0.1"
  mini2: "10.10.0.2"
  mini3: "10.10.0.2"
  pi4: "10.10.0.1"
  macbook: "10.10.0.1"
  pi5: "192.168.50.1"
  win: "10.10.0.1"
  ipad: "172.20.10.3"     # iPad via WiFi

port:
  default: 65432
  mini1: 65432
  mini2: 65432
  mini3: 65432
  pi4: 65432
  macbook: 65432
  pi5: 65432
  win: 65432
  ipad: 8000              # iPad uses port 8000 (port forwarding)

web_interface:
  api_port: 8080          # FastAPI backend port
  frontend_port: 5050     # Frontend HTTP server port

# Network buffer sizes (in MB) - device-specific optimizations
buffer_size:
  mini1: 8                # Mac Mini - 8MB (Thunderbolt 40Gbps)
  mini2: 8                # Mac Mini - 8MB
  mini3: 8                # Mac Mini - 8MB
  macbook: 8              # MacBook - 8MB
  pi4: 2                  # Raspberry Pi 4 - 2MB (Gigabit Ethernet)
  pi5: 2                  # Raspberry Pi 5 - 2MB (Gigabit Ethernet)
  win: 1                  # Windows 11 - 1MB (conservative)

# Network metrics tracking
track_network_metrics: true  # Enable bandwidth, latency, buffer usage tracking
metrics_log_interval: 50     # Log network metrics every N steps

num_workers: 2
server: mini2
timeout: 0.1                 # Server timeout for gradient collection (seconds)
seed: 42
```

#### Parameters

**Web Interface:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_port` | int | FastAPI backend port (default: 8080) |
| `frontend_port` | int | Frontend HTTP server port (default: 5050) |

**Network Configuration:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `host_ip` | dict | Device hostname to IP address mapping |
| `port` | dict | Per-device port configuration |
| `buffer_size` | dict | Device-specific buffer sizes (MB) |
| `track_network_metrics` | bool | Enable bandwidth/latency monitoring |
| `metrics_log_interval` | int | Metrics logging frequency (steps) |
| `num_workers` | int | Number of worker nodes |
| `server` | str | Server hostname |
| `timeout` | float | Gradient collection timeout (seconds) |
| `seed` | int | Random seed |

**Buffer Size Recommendations:**

| Device Type | Buffer Size | Network Capability |
|-------------|-------------|--------------------|
| Mac mini M4 | 8 MB | Thunderbolt 40Gbps |
| MacBook | 8 MB | Thunderbolt 40Gbps |
| Raspberry Pi 4/5 | 2 MB | Gigabit Ethernet |
| Windows 11 | 1 MB | Conservative default |

---

## Network Topology

### Standard Configuration (Training)

Typical setup for distributed training across Mac minis and Raspberry Pis.

**Thunderbolt Fabric** (High-speed inter-Mac):
```
Mac mini 1 (SERVER) — 10.10.0.1  ─┐
Mac mini 2 (WORKER) — 10.10.0.2  ─┼─ Thunderbolt Bridge
Mac mini 3 (WORKER) — 10.10.0.3  ─┘
```

**Ethernet Edge Links** (Pi connectivity):
```
Pi 5 (192.168.50.2) ──── Mac mini 1 (192.168.50.1)
Pi 4 (192.168.51.4) ──── Mac mini 3 (192.168.51.2)
```

**Design Principle**: One subnet per physical link. Pis route to Thunderbolt network via their connected Mac.

### Hybrid Configuration (iPad + Mac Mini Inference)

Optimized setup for distributed GPT-2 inference with CoreML acceleration on iPad.

**Network Layout:**
```
┌─────────────────────────────────────────────────┐
│  Mac mini M4 #1    Mac mini M4 #2               │
│  (Rank 0)          (Rank 2)                     │
│  10.10.0.1         10.10.0.2                    │
│      └──────────────┘                           │
│      Thunderbolt 40Gbps                         │
└─────────────────────────────────────────────────┘
                 │
                 │ WiFi Network
                 │
         ┌───────┴────────┐
         │                │
    iPad (Rank 1)    MacBook (Controller)
    172.20.10.3      10.10.0.3
    CoreML Layers    FastAPI + Frontend
```

**Layer Distribution:**
- **Rank 0** (Mac mini M4 #1): Layers 0-3 (embedding + first 4 transformer blocks)
- **Rank 1** (iPad CoreML): Layers 4-7 (middle 4 transformer blocks, Neural Engine)
- **Rank 2** (Mac mini M4 #2): Layers 8-11 (final 4 transformer blocks + LM head)

**Controller:** MacBook runs FastAPI backend and React frontend for user interaction.

---

## Distributed Inference Deployment

### Single-Script Launch

Deploy the complete distributed inference stack with one command:

```bash
cd ~/Desktop/smolcluster/scripts/inference
./launch_mp_inference.sh
```

**This script automatically:**
1. Starts inference server on Rank 0 (Mac mini)
2. SSH into workers and starts processes on Rank 1, 2 (iPad, Mac mini)
3. Launches FastAPI backend on controller (MacBook)
4. Starts frontend server on `http://localhost:3000`
5. Configures all network connections and layer distributions

**No manual configuration needed** - the script reads from `cluster_config_inference.yaml` and `model_config_inference.yaml`.

### Architecture Components

#### React Frontend

**Location:** `src/smolcluster/chat/frontend/`

**Features:**
- Real-time chat interface with streaming token display
- Parameter configuration (max tokens, temperature, top-p, top-k)
- Message history and session management
- Auto-reconnect on network interruptions

#### FastAPI Backend

**Location:** `src/smolcluster/chat/backend/api.py`

**Features:**
- RESTful API for inference requests
- WebSocket support for real-time token streaming
- Health monitoring and reconnection handling

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Submit inference requests |
| `/config` | GET | Retrieve active model configuration |
| `/health` | GET | Health check endpoint |
| `/reconnect` | POST | Reconnect to inference server |

**Communication Flow:**
```
User (Browser) → FastAPI Backend → Inference Server (Rank 0)
                                          ↓
                                    Worker 1 (Rank 1)
                                          ↓
                                    Worker 2 (Rank 2)
                                          ↓
                   ← Streaming Tokens ←  ←
```

---

## Command-Line Overrides

Override configuration parameters at runtime without modifying YAML files:

```bash
# Override GPT training parameters
uv run python src/smolcluster/train.py \
  --override training.batch_size=16 \
  --override training.epochs=5 \
  --override training.learning_rate=3e-4

# Use custom config file
uv run python src/smolcluster/train.py \
  --config path/to/custom_config.yaml
```
