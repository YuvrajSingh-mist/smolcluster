# Model Parallelism Inference

This guide explains how to deploy distributed GPT inference using Model Parallelism across multiple nodes in the SmolCluster.

## Overview

Model Parallelism enables large language model inference by splitting model layers across multiple workers, allowing inference on models that exceed a single device's memory capacity. The system uses a leader-worker architecture where activations are sequentially forwarded through distributed transformer layers.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Client    │─────▶│    Server    │─────▶│   Worker 1   │─────▶│   Worker 2   │
│  (FastAPI)  │      │  (Rank 0)    │      │  (Rank 1)    │      │  (Rank 2)    │
│             │◀─────│  Layers 0-4  │      │  Layers 5-9  │      │ Layers 10-11 │
└─────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                            │                      │                      │
                            └──────────────────────┴──────────────────────┘
                                    Activations Flow (CPU Tensors)
```

**Components:**
- **Server (Rank 0)**: Manages client connections, coordinates workers, handles tokenization, and sampling
- **Workers (Rank 1+)**: Process activations through their assigned model layers
- **FastAPI Backend**: REST API for chat applications
- **Frontend**: Web interface for user interaction

## Configuration

### Model Configuration

Edit `src/smolcluster/configs/model_parallelism/model_config.yaml`:

```yaml
causal_gpt2:
  hf_model_name: "gpt2"
  weights_model_name: "gpt2"  # Auto-downloads from HuggingFace
  num_layers: 12
  num_nodes: 3  # Server + 2 workers
  max_new_tokens: 50
  
  # Decoding strategies
  active_decoding_strategy: "top_k"
  decoding_strategies:
    greedy: {}
    sampling:
      temperature: 1.0
    top_p:
      temperature: 1.0
      p: 0.9
    top_k:
      temperature: 1.0
      k: 40
```

**Key Parameters:**
- `num_nodes`: Total nodes (server + workers)
- `num_layers`: Total transformer layers to distribute
- `weights_model_name`: Model identifier for auto-downloading safetensors
- `active_decoding_strategy`: Default strategy (overridable via API)

### Cluster Configuration

Edit `src/smolcluster/configs/cluster_config_syncps.yaml`:

```yaml
host_ip:
  mini1: "10.10.0.1"    # Server
  mini2: "10.10.0.2"    # Worker 1
  pi5: "192.168.50.2"   # Worker 2

port: 65432
num_workers: 2
workers: [mini2, pi5]
server: mini1
```

### Model Weights

Weights are automatically downloaded on first run from HuggingFace. Supported models:

```yaml
# src/smolcluster/configs/model_weights.yaml
models:
  gpt2:
    url: "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    filename: "gpt2.safetensors"
  gpt2-medium:
    url: "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors"
    filename: "gpt2-medium.safetensors"
```

**Storage:** Downloaded to `src/data/<filename>`

## Deployment

### 1. Start Inference Server

On the server node (e.g., mini1):

```bash
cd ~/Desktop/smolcluster
python -m smolcluster.algorithms.ModelParallelism.inference.server
```

**Server Output:**
```
[LEADER] Server will bind to IP: 0.0.0.0, Port: 65432
[LEADER] Checking for model weights (gpt2)...
[LEADER] Model weights ready at: /Users/.../src/data/gpt2.safetensors
[LEADER] Model initialized on device: mps
[LEADER] Loading server's share of model layers (rank 0)...
[LEADER] Server loaded 4 layers
[LEADER] Server listening on 0.0.0.0:65432
```

### 2. Start Workers

On each worker node (e.g., mini2, pi5):

```bash
cd ~/Desktop/smolcluster
python -m smolcluster.algorithms.ModelParallelism.inference.worker <RANK> <HOSTNAME>
```

**Example:**
```bash
# Worker 1 on mini2
python -m smolcluster.algorithms.ModelParallelism.inference.worker 1 mini2

# Worker 2 on pi5
python -m smolcluster.algorithms.ModelParallelism.inference.worker 2 pi5
```

**Worker Output:**
```
[WORKER-1] Worker 1 starting. Connecting to server at 10.10.0.1:65432
[WORKER-1] Model initialized on device: mps
[WORKER-1] Checking for model weights (gpt2)...
[WORKER-1] Model weights ready at: /Users/.../src/data/gpt2.safetensors
[WORKER-1] Loaded 4 layers for worker 1
[WORKER-1] Connected to server at 10.10.0.1:65432 on attempt 1
[WORKER-1] Registering as worker 1 with server...
[WORKER-1] Received start_inference command from server.
[WORKER-1] Waiting for generation requests...
```

### 3. Start FastAPI Backend

On any node with network access to the server:

```bash
cd ~/Desktop/smolcluster
python -m smolcluster.chat.backend.api
```

**Configuration in `chat/backend/api.py`:**
```python
SERVER_HOST = "10.10.0.1"  # Update to server IP
SERVER_PORT = 65432
```

**API Endpoints:**
- `POST /chat`: Submit inference requests
- `GET /config`: Retrieve active model configuration
- `GET /health`: Health check
- `POST /reconnect`: Reconnect to inference server

### 4. Access Frontend

Open `src/smolcluster/chat/frontend/index.html` in a browser, or serve it:

```bash
cd src/smolcluster/chat/frontend
python -m http.server 8080
# Access at http://localhost:8080
```

**Frontend Features:**
- Real-time chat interface
- Parameter display (tokens, temperature, top-p, top-k, strategy)
- Message history
- Auto-reconnect on disconnect

## API Usage

### Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is distributed computing?",
    "max_tokens": 30,
    "temperature": 0.8,
    "top_k": 40
  }'
```

**Response:**
```json
{
  "response": "Distributed computing is a field of computer science that studies distributed systems...",
  "config": {
    "tokens_generated": 30,
    "temperature": 0.8,
    "top_k": 40,
    "decoding_strategy": "top_k"
  }
}
```

### Get Configuration

```bash
curl http://localhost:8000/config
```

**Response:**
```json
{
  "max_tokens": 50,
  "temperature": 1.0,
  "top_k": 40,
  "active_strategy": "top_k"
}
```

## Inference Flow

1. **Client Sends Prompt** → FastAPI backend via REST
2. **Backend Forwards** → Server socket connection
3. **Server Tokenizes** → Converts text to input IDs
4. **Server Processes** → Forward through rank 0 layers
5. **Server → Worker 1** → Sends activations (CPU tensors)
6. **Worker 1 Processes** → Forward through rank 1 layers
7. **Worker 1 → Worker 2** → Sends activations
8. **Worker 2 → Server** → Returns final activations
9. **Server Samples Token** → Using configured decoding strategy
10. **Repeat Steps 4-9** → For `max_new_tokens` iterations
11. **Server Decodes** → Token IDs to text (excluding prompt)
12. **Backend Returns** → Generated text to client

## Decoding Strategies

### Greedy Decoding
```python
# Always selects highest probability token
decoding_strategy: "greedy"
```

### Sampling with Temperature
```python
decoding_strategy: "sampling"
temperature: 1.0  # Higher = more random
```

### Top-p (Nucleus) Sampling
```python
decoding_strategy: "top_p"
temperature: 1.0
p: 0.9  # Cumulative probability threshold
```

### Top-k Sampling
```python
decoding_strategy: "top_k"
temperature: 1.0
k: 40  # Consider top-k most probable tokens
```

## Layer Distribution

Layers are automatically distributed based on `num_nodes`. Example for GPT-2 (12 layers, 3 nodes):

| Node | Rank | Layers | Count |
|------|------|--------|-------|
| Server (mini1) | 0 | Embedding + 0-3 | 4 |
| Worker 1 (mini2) | 1 | 4-7 | 4 |
| Worker 2 (pi5) | 2 | 8-11 | 4 |

**Distribution Logic:**
```python
# Automatic distribution in utils/layers.py
layers_per_node = num_layers // num_nodes
start_layer = rank * layers_per_node
end_layer = (rank + 1) * layers_per_node
```

## Troubleshooting

### Worker Cannot Connect

**Symptom:** Worker times out connecting to server

**Solution:**
- Verify server IP in `cluster_config_syncps.yaml`
- Check firewall rules: `sudo ufw allow 65432/tcp`
- Test connectivity: `ping <server_ip>` from worker
- Ensure server is listening: Check server logs for "Server listening"

### Model Weights Not Found

**Symptom:** `FileNotFoundError: model.safetensors`

**Solution:**
- Check internet connection
- Verify `model_weights.yaml` has correct URLs
- Manually download: `python -m smolcluster.utils.model_downloader gpt2`
- Check disk space in `src/data/`

### Memory Errors

**Symptom:** `RuntimeError: CUDA out of memory` or similar

**Solution:**
- Increase `num_nodes` to distribute layers across more devices
- Use smaller model: Change `weights_model_name` to `gpt2` instead of `gpt2-xl`
- Reduce batch size (inference uses batch_size=1 by default)

### Incorrect Generation Output

**Symptom:** Gibberish or repetitive output

**Solution:**
- Check temperature: Lower values (0.7-0.9) for coherent text
- Adjust top_k or top_p for better diversity
- Verify all workers loaded correct layer ranges (check logs)
- Ensure all nodes use same `weights_model_name`

### API Connection Refused

**Symptom:** FastAPI cannot reach inference server

**Solution:**
- Update `SERVER_HOST` in `chat/backend/api.py`
- Ensure inference server started before FastAPI
- Check server socket is bound: `netstat -an | grep 65432`
- Verify CORS settings for cross-origin requests

## Performance Considerations

**Latency:** Each layer transition adds network overhead. For optimal performance:
- Use high-bandwidth connections (Thunderbolt > Ethernet > WiFi)
- Minimize `num_nodes` while fitting memory constraints
- Consider layer-specific distribution for balanced compute

**Throughput:** Single-token generation is sequential. For batch inference:
- Modify server to batch prompts
- Process multiple tokens in parallel (future enhancement)

**Memory:** Each worker stores:
- Assigned model layers (~350MB per 4 GPT-2 layers)
- Intermediate activations (~1-10MB depending on sequence length)

## Advanced Configuration

### Custom Layer Distribution

Override default distribution in `utils/layers.py`:

```python
# Example: Asymmetric distribution
layer_mapping = {
    0: [0, 1, 2],      # Server: 3 layers
    1: [3, 4, 5, 6],   # Worker 1: 4 layers
    2: [7, 8, 9, 10, 11]  # Worker 2: 5 layers
}
```

### Multi-Model Support

Add models to `model_config.yaml`:

```yaml
gpt2_large:
  hf_model_name: "gpt2-large"
  weights_model_name: "gpt2-large"
  num_layers: 36
  num_nodes: 6  # More layers need more nodes
```

### Custom Decoding Strategy

Add to `utils/decoding.py`:

```python
def custom_sampling(logits, temperature, **kwargs):
    # Your custom sampling logic
    pass
```

Register in `sample_next_token()` function.

## Production Deployment

### Security
- Use TLS for API endpoints (nginx reverse proxy)
- Implement authentication (JWT tokens)
- Restrict CORS origins in FastAPI
- Use private network for inter-node communication

### Monitoring
- Enable centralized logging (Promtail → Loki → Grafana)
- Track latency per token generation
- Monitor memory usage per worker
- Set up alerts for worker disconnections

### Scaling
- Horizontal: Add more workers for larger models
- Vertical: Use GPU workers for faster compute
- Load balancing: Deploy multiple server instances

## References

- [Model Parallelism Training](./setup_cluster.md) - Distributed training guide
- [Logging Setup](./logging.md) - Centralized logging configuration
- [HuggingFace Models](https://huggingface.co/models) - Available pretrained models
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Original architecture
