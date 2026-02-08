# Training Guide

## Training Algorithms

SmolCluster implements three distributed training paradigms:

### Elastic Distributed Parallelism (EDP)

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

## Algorithm Comparison

| Feature | EDP | SyncPS | Model Parallelism |
|---------|-----|--------|-------------------|
| **Synchronization** | Asynchronous | Synchronous | Sequential |
| **Gradient Staleness** | Tolerates stale | Fresh only | N/A |
| **Barrier Points** | None | Every step | Per layer |
| **Throughput** | Highest | Medium | Lowest |
| **Convergence** | Slower | Faster | N/A (inference) |
| **Fault Tolerance** | Best | Good | Poor |
| **Memory Efficiency** | Low | Low | High |
| **Network Efficiency** | Quantization supported | Raw gradients | Activations only |

## How Each Algorithm Works

### EDP (Elastic Distributed Parallelism)

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
# EDP
cd src/smolcluster/algorithms/EDP
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
Example: `EDP-worker-macbook_rank1_lr0.001_bs32`

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

## Troubleshooting

### Connection Issues

**Symptom:** Workers can't connect to server

**Solutions:**
```bash
# Verify network connectivity
ping 10.10.0.1

# Check port availability
nc -zv 10.10.0.1 65432

# Verify firewall rules
sudo ufw allow 65432/tcp  # Linux
# macOS: System Preferences → Security → Firewall → Options

# Ensure server started before workers
tmux attach -t mp_server  # Check server logs
```

### Training Issues

**Gradients exploding:**
```yaml
# nn_config.yaml or gpt_config.yaml
gradient_clipping:
  enabled: true
  max_norm: 1.0
```

**Loss not decreasing:**
- Reduce learning rate: `--override training.learning_rate=1e-4`
- Increase batch size: `--override training.batch_size=64`
- Check data loading: Verify dataset partitioning is correct

**SyncPS stragglers:**
```yaml
# cluster_config_syncps.yaml
timeout: 1.0  # Increase timeout for slow workers
```

**EDP staleness issues:**
```yaml
# cluster_config_edp.yaml
worker_update_interval: 5  # Decrease interval for fresher weights
```

### W&B Issues

**Authentication:**
```bash
# Set API key
export WANDB_API_KEY=your_api_key_here
wandb login
```

**Offline mode:**
```bash
export WANDB_MODE=offline
```

**Missing metrics:**
- Check `track_gradients: true` in config for gradient norms
- Verify network connectivity to wandb.ai
- Review W&B run names match expected format

### Memory Issues

**Out of memory errors:**
- Reduce batch size
- Enable gradient checkpointing (for GPT)
- Use model parallelism for large models
- Reduce model dimensions

### Performance Optimization

**Slow training:**
- Use Model Parallelism for large models
- Enable gradient quantization (EDP only)
- Reduce `eval_steps` frequency
- Increase `worker_update_interval` (EDP)
- Optimize network topology (use Thunderbolt for high-bandwidth links)

**Network bottleneck:**
```yaml
# cluster_config_edp.yaml
use_quantization: true  # Enable 8-bit gradient compression
```
