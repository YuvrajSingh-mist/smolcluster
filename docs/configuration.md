# Configuration Guide

SmolCluster uses YAML configuration files located in `src/smolcluster/configs/` to manage cluster topology, model architecture, and training parameters.

## Cluster Configuration

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

**Configuration Parameters:**
- `host_ip`: Maps hostnames to their network addresses
- `port`: TCP port for socket communication (default: 65432)
- `num_workers`: Number of worker nodes expected
- `workers`: List of worker hostnames (must match keys in `host_ip`)
- `server`: Server hostname (must match key in `host_ip`)
- `worker_update_interval`: How often workers pull latest weights from server (in training steps)
- `use_quantization`: Enable 8-bit quantization for gradient communication
- `seed`: Random seed for reproducible data partitioning

### cluster_config_syncps.yaml (Synchronous Parameter Server)

```yaml
host_ip:
  mini1: "10.10.0.1"
  macbook: "10.10.0.1"
  pi5: "192.168.50.1"
  win: "10.10.0.1"

port: 65432
num_workers: 3
workers: [pi5, win, macbook]
server: mini1
worker_update_interval: 4     # Steps between full weight synchronization
timeout: 0.1                  # Server timeout for gradient collection (seconds)
seed: 42
```

**Additional Parameters:**
- `worker_update_interval`: How often workers pull and apply Polyak-averaged weights
- `timeout`: How long server waits for gradients before proceeding (for handling stragglers)

### cluster_config_mp.yaml (Model Parallelism)

```yaml
host_ip:
  mini1: "10.10.0.1"
  mini2: "10.10.0.2"
  mini3: "10.10.0.3"

port: 65432
num_workers: 2                # Number of workers (total nodes - 1)
workers: [mini2, mini3]
server: mini1
seed: 42
timeout: 300                  # Timeout for inference requests
```

## Model Configuration

### nn_config.yaml (Simple Neural Network)

```yaml
batch_size: 32
learning_rate: 0.001
num_epochs: 2
eval_steps: 200             # Evaluate every N steps
track_gradients: true       # Log gradient norms to W&B
polyak_alpha: 0.5          # Polyak averaging coefficient
gradient_clipping:
  enabled: true
  max_norm: 1.0

model:
  type: SimpleNN
  input_dim: 784
  hidden: 128
  out: 10

dataset:
  name: MNIST
```

**Parameters:**
- `batch_size`: Batch size per worker
- `learning_rate`: Learning rate for gradient descent
- `num_epochs`: Total training epochs
- `eval_steps`: Evaluation frequency
- `track_gradients`: Enable per-layer gradient norm logging
- `polyak_alpha`: Blending factor for Polyak averaging (0.0-1.0)
- `gradient_clipping`: Prevent exploding gradients

### gpt_config.yaml (GPT Language Model)

```yaml
model:
  model_dim: 256          # Model dimension
  num_layers: 6           # Transformer layers
  num_heads: 4            # Attention heads
  ff_dim: 1024            # Feed-forward dimension
  dropout: 0.1
  max_seq_len: 128        # Maximum sequence length

training:
  batch_size: null        # Auto: 32 (L=128) or 16 (L=256)
  epochs: null            # Auto: 30 (wikitext) or 20 (SNLI)
  learning_rate: 6e-4
  weight_decay: 0.01      # AdamW weight decay
  grad_clip_norm: 1.0

lr_schedule:
  warmup_iters: 100
  min_lr: 6e-5

data:
  tokenizer: "openai-community/gpt2"

logging:
  project_name: "smolcluster-gpt-wikitext2"
```

**Auto-calculated Parameters:**
- `batch_size`: Automatically set based on sequence length
- `epochs`: Automatically set based on dataset

## Model Parallelism Configuration

### model_config_inference.yaml

```yaml
causal_gpt2:
  hf_model_name: "openai-community/gpt2"
  weights_model_name: "gpt2"
  num_nodes: 3              # Number of nodes for model parallelism
  num_layers: 12            # Total transformer layers
  max_new_tokens: 256       # Maximum tokens to generate
  
  active_decoding_strategy: "top_k"
  
  decoding_strategies:
    top_p:
      temperature: 1.0
      p: 0.9
    top_k:
      temperature: 1.0
      k: 40
    top_k_top_p:
      temperature: 1.0
      k: 50
      p: 0.9
```

**Decoding Strategy Parameters:**
- `active_decoding_strategy`: Which strategy to use by default
- `temperature`: Sampling temperature (higher = more random)
- `p`: Top-p (nucleus) sampling threshold
- `k`: Top-k sampling threshold

### cluster_config_inference.yaml

```yaml
host_ip:
  mini1: "10.10.0.1"
  mini2: "10.10.0.2"
  mini3: "10.10.0.3"

port: 65432
num_workers: 2
workers: [mini2, mini3]
server: mini1
timeout: 300

web_interface:
  api_port: 8080          # FastAPI backend port
  frontend_port: 5050     # Frontend HTTP server port
```

**Web Interface:**
- `api_port`: Port for FastAPI backend
- `frontend_port`: Port for static HTML frontend

## Network Topology

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

## Command-Line Overrides

Override config parameters from the command line:

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
