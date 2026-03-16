# Training Guide

## Table of Contents

- [Training Algorithms](#training-algorithms)
  - [Fully Sharded Data Parallel](#fsdp-fully-sharded-data-parallel)
  - [Classic Data Parallelism (ClassicDP)](#classic-data-parallelism-classicdp)
  - [Elastic Distributed Parallelism](#elastic-distributed-parallelism-edp)
  - [Synchronous Parameter Server (SyncPS)](#synchronous-parameter-server-syncps)
  - [Expert Parallelism (EP)](#expert-parallelism-ep)
  - [Model Parallelism (MP)](#model-parallelism-mp)
- [Algorithm Comparison](#algorithm-comparison)
- [How Each Algorithm Works](#how-each-algorithm-works)
  - [Fully Sharded Data Parallel](#fsdp-fully-sharded-data-parallel-1)
  - [ClassicDP (Classic Data Parallelism)](#classicdp-classic-data-parallelism)
  - [Elastic Distributed Parallelism](#edp-elastic-distributed-parallelism)
  - [SyncPS (Synchronous Parameter Server)](#syncps-synchronous-parameter-server)
  - [EP (Expert Parallelism)](#ep-expert-parallelism)
  - [Model Parallelism](#model-parallelism)
- [Usage Examples](#usage-examples)
  - [MNIST Training (Simple NN)](#mnist-training-simple-nn)
  - [GPT Training (Language Model)](#gpt-training-language-model)
  - [MoE Training (Expert Parallelism)](#moe-training-expert-parallelism)
  - [GPT Inference (Model Parallelism)](#gpt-inference-model-parallelism)
- [Monitoring Training](#monitoring-training)
  - [Console Logs](#console-logs)
  - [Weights & Biases (W&B)](#weights--biases-wb)
  - [Grafana (Centralized Logging)](#grafana-centralized-logging)

---

## Training Algorithms

SmolCluster implements multiple distributed training paradigms:

### Fully Sharded Data Parallel

**ZeRO-optimized data parallelism with optimizer state partitioning**

- Configurable ZeRO stages (Stage 0: Optimizer, Stage 1: +Gradient, Stage 2: +Parameter Partitioning)
- All-to-all gradient communication (ring all-reduce topology)
- Reduced memory footprint through state partitioning
- Bandwidth-optimized weight broadcasting (only owned parameters)
- Configurable staleness bound for async flexibility
- Real-time staleness metrics tracked in WandB

**Best for:** Memory-constrained setups, large models, efficient bandwidth usage

**Launch:**
```bash
bash scripts/launch_fsdp_train_gpt.sh
```

**Configuration:**
```yaml
# In cluster_config_fsdp.yaml
fsdp_stage: 0  # 0=Optimizer, 1=+Gradient, 2=+Parameter partitioning
staleness_bound: 5  # Allow workers to be 5 steps apart (0 = strict sync)
```

**Fully Sharded Data Parallel Stages:**
- `fsdp_stage: 0` - ZeRO-0 (optimizer state partitioning, ~1/N optimizer memory)
- `fsdp_stage: 1` - ZeRO-1 (optimizer + gradient partitioning, ~1/N optimizer + gradient memory)
- `fsdp_stage: 2` - ZeRO-2 (optimizer + gradient + parameter partitioning, ~1/N total memory)

### Classic Data Parallelism (ClassicDP)

**All-Reduce based data parallelism with bounded staleness**

- All-to-all gradient communication (ring all-reduce topology)
- Workers exchange gradients directly (no parameter server)
- Configurable staleness bound for async flexibility
- Real-time staleness metrics tracked in WandB
- Automatic cleanup of stale gradients beyond bound

**Best for:** Balanced clusters, moderate network latency, fine-grained async control

**Launch:**
```bash
bash scripts/launch_dp_train_gpt.sh
```

**Configuration:**
```yaml
# In cluster_config_classicdp.yaml
staleness_bound: 5  # Allow workers to be 5 steps apart (0 = strict sync)
```

**Staleness Modes:**
- `staleness_bound: 0` - Strict synchronous (all workers at same step)
- `staleness_bound: K` - Bounded async (workers can drift up to K steps)

### Elastic Distributed Parallelism

**Asynchronous data parallelism with stale gradient handling**

- Workers operate independently without synchronization barriers
- Server accepts gradients from any model version (tolerates staleness)
- Workers periodically pull latest weights from server
- Optional gradient quantization for bandwidth efficiency

**Best for:** Heterogeneous clusters, high-latency networks, fault tolerance

**Launch:**
```bash
bash scripts/launch_edp_train_gpt.sh
```

### Synchronous Parameter Server (SyncPS)

**Synchronous data parallelism with barrier-based coordination**

- All workers must complete each step before proceeding
- Server only accepts fresh gradients from current training step
- Polyak-averaged weights distributed periodically
- Lower latency requirements

**Best for:** Homogeneous clusters, low-latency networks, faster convergence

**Launch:**
```bash
bash scripts/launch_syncps_train_gpt.sh
```

### Expert Parallelism (EP)

**Mixture-of-Experts training with experts sharded across nodes**

- Routes tokens to top-k experts for sparse activation
- Each node hosts a subset of experts (expert sharding)
- Collective all-to-all communication for token dispatch and combine
- Scales expert capacity with lower per-token compute than dense models

**Best for:** MoE models (e.g., Mixtral-style), scaling parameter count efficiently, heterogeneous clusters with expert placement control

**Launch:**
```bash
bash scripts/training/launch_ep_train_moe.sh
```

### Model Parallelism (MP)

**Layer-wise model distribution across nodes**

- Model layers split across multiple devices
- Sequential activation passing between nodes
- Enables training of models larger than single-device memory
- Supports distributed inference with streaming

**Best for:** Large models, inference serving, memory-constrained devices

**Launch:**
```bash
# Training
bash scripts/launch_mp_train_gpt.sh

# Inference
bash scripts/inference/launch_mp_inference.sh
bash scripts/inference/launch_api.sh
```

## How Each Algorithm Works

### Fully Sharded Data Parallel

**Stage 0: All-Reduce Mode**
1. **All-to-All Topology Setup**: Workers form a fully connected graph
2. **Local Gradient Computation**: Each worker computes gradients on its data shard
3. **All-Gather Phase**: Each worker broadcasts gradients to all peers
4. **Reduce Phase**: Workers average all received gradients
5. **Model Update**: Apply averaged gradients to full model replica
6. **Weight Broadcast**: Synchronize model weights across all workers

**Stage 1: ZeRO Optimizer Partitioning**
1. **Layer Partitioning**: Each worker owns specific model layers
2. **Optimizer Creation**: Worker creates optimizer only for owned parameters (~1/N memory)
3. **Forward/Backward**: Each worker runs full model on local data
4. **All-Reduce Gradients**: Average gradients across all workers
5. **Partial Update**: Worker's optimizer updates only its owned parameters
6. **Optimized Broadcast**: Each worker broadcasts only its updated parameters (not full model)
7. **Weight Merging**: Workers merge received parameters into full model

**ZeRO Stage 1 Benefits:**
- Memory savings: Optimizer states distributed across workers (1/N per worker)
- Bandwidth optimization: Only owned parameters broadcasted (1/N traffic per worker)
- Full model maintained: Each worker has complete model for training
- Scalability: Memory and bandwidth scale linearly with worker count

**Bounded Staleness:**
- `staleness_bound = 0`: All workers at exact same step (strict sync)
- `staleness_bound = K`: Workers can be up to K steps apart (bounded async)
- Training stops if any gradient/weight update exceeds the bound

### ClassicDP (Classic Data Parallelism)

1. **All-to-All Topology Setup**: Workers form a fully connected graph
2. **Local Gradient Computation**: Each worker computes gradients on its data shard
3. **All-Gather Phase**:
   - Each worker broadcasts its gradients to all peers
   - Workers buffer incoming gradients by step number
   - Staleness check: reject gradients beyond bound (if configured)
4. **Reduce Phase**:
   - Workers average all received gradients
   - Apply averaged gradients to local model
5. **Scatter-Reduce Phase** (optional, for learning):
   - Workers broadcast averaged gradients back to peers
   - Used for validation and monitoring
6. **Staleness Tracking**:
   - Track step differences for all received gradients
   - Log to WandB: avg/max step diff, stale gradient count
   - Auto-cleanup gradients beyond staleness window

**Bounded Staleness:**
- `staleness_bound = 0`: All workers must be at exact same step (strict sync)
- `staleness_bound = K`: Workers can be up to K steps apart (bounded async)
- Training stops if any gradient exceeds the bound

### Elastic Distributed Parallelism

1. **Server Initialization**: Parameter server starts and waits for workers
2. **Worker Registration**: Each worker connects and receives initial weights
3. **Asynchronous Training Loop**:
   - Worker computes gradients on local batch
   - Worker pushes gradients to server (non-blocking)
   - Server accumulates and applies gradients
   - Worker periodically pulls updated weights (every N steps)
4. **No Synchronization**: Workers never wait for each other
5. **Optional Quantization**: 8-bit gradient compression reduces bandwidth

### SyncPS (Synchronous Parameter Server)

1. **Server Initialization**: Parameter server starts with barrier tracking
2. **Worker Registration**: Workers connect and sync initial state
3. **Synchronous Training Loop**:
   - All workers compute gradients on local batch
   - Workers send gradients to server
   - Server waits for all gradients (with timeout for stragglers)
   - Server averages gradients and updates model
   - Workers pull Polyak-averaged weights periodically
4. **Barrier Synchronization**: Step N+1 starts only after all workers finish step N
5. **Polyak Averaging**: Smooth weight updates using exponential moving average

### EP (Expert Parallelism)

1. **Expert Partitioning**: Experts are distributed across nodes based on configured ranks
2. **Token Routing**: Router selects top-k experts for each token
3. **All-to-All Dispatch**: Tokens are exchanged so each expert receives its assigned tokens
4. **Local Expert Compute**: Each node computes only for its local experts
5. **All-to-All Combine**: Expert outputs are returned and merged in original token order
6. **Backpropagation**: Gradients flow through router and expert paths, then optimizer updates local expert weights

### Model Parallelism

1. **Layer Distribution**: Model split across nodes (e.g., layers 0-3 on node 0, 4-7 on node 1)
2. **Forward Pass**:
   - Node 0 processes input through its layers
   - Activations sent to Node 1
   - Node 1 processes through its layers
   - Final activations returned
3. **Backward Pass** (training):
   - Gradients flow backward through nodes
   - Each node updates its layers
4. **Sequential Dependency**: Each node waits for previous node's output

## Usage Examples

### MNIST Training (Simple NN)

```bash
# Elastic Distributed Parallelism
cd src/smolcluster/algorithms/Elastic Distributed Parallelism
uv run server.py  # On server node
uv run worker.py 1 macbook  # On worker node

# SyncPS
cd src/smolcluster/algorithms/DataParallelism/SynchronousPS
uv run server.py
uv run worker.py 1 pi5
```

### GPT Training (Language Model)

```bash
# Automated launch (recommended)
bash scripts/launch_edp_train_gpt.sh

# Manual override
uv run python src/smolcluster/train.py \
  --override training.batch_size=16 \
  --override training.learning_rate=3e-4
```

### MoE Training (Expert Parallelism)

```bash
# Automated launch (recommended)
bash scripts/training/launch_ep_train_moe.sh

# Optional: resume from checkpoint
bash scripts/training/launch_ep_train_moe.sh --resume-checkpoint checkpoints/latest.pt
```

### GPT Inference (Model Parallelism)

```bash
# Terminal 1: Start distributed inference server
bash scripts/inference/launch_mp_inference.sh

# Terminal 2: Start API and web interface
bash scripts/inference/launch_api.sh

# Access at http://localhost:5050
```

## Monitoring Training

### Console Logs

Each tmux pane shows real-time training progress:
- Training/validation loss
- Step timing and throughput
- Gradient norms (if enabled)
- Connection status

**Navigation:**
- `Ctrl+B` then arrow keys: Switch panes
- `Ctrl+B` then `[`: Scroll mode
- `Ctrl+B` then `d`: Detach session
- `tmux attach`: Reattach to session

### Weights & Biases (W&B)

Automatic experiment tracking with detailed metrics:

**Dashboard:** [wandb.ai](https://wandb.ai)  
**Project:** `smolcluster` or `smolcluster-gpt-wikitext2`

**Metrics Logged:**
- Training loss per step
- Validation loss and accuracy
- Learning rate schedule
- Per-layer gradient norms (if `track_gradients: true`)
- Step timing and tokens/second
- Hardware utilization

**Run Naming:**  
Format: `{Algorithm}-{role}-{hostname}_rank{X}_lr{Y}_bs{Z}`  
Example: `Elastic Distributed Parallelism-worker-macbook_rank1_lr0.001_bs32`

### Grafana (Centralized Logging)

Real-time log aggregation from all nodes:

**Dashboard:** [http://localhost:3000](http://localhost:3000)  
**Credentials:** admin/admin

**Query Examples:**
```
{job="smolcluster-worker"}           # All worker logs
{job="smolcluster-server"}           # Server logs
{host="worker-rank0-mini2"}          # Specific worker
{level="ERROR"}                      # Only errors
{level="INFO"} |= "Epoch"            # Training progress
```

See [logging.md](logging.md) for full setup instructions.
