# Model Parallelism Inference

This guide explains how to deploy distributed GPT inference using Model Parallelism across multiple nodes in the Smolcluster.

## Table of Contents

- [Overview](#overview)
- [Supported Deployment Configurations](#supported-deployment-configurations)
  - [Configuration 1: Mac Mini Cluster (CPU-based)](#configuration-1-mac-mini-cluster-cpu-based)
  - [Configuration 2: iPad + Mac Mini Hybrid Cluster (CoreML + CPU)](#configuration-2-ipad--mac-mini-hybrid-cluster-coreml--cpu)
- [Architecture](#architecture)
  - [Standard Configuration (Mac Mini Cluster)](#standard-configuration-mac-mini-cluster)
  - [Hybrid Configuration (iPad + Mac Mini with MacBook Controller)](#hybrid-configuration-ipad--mac-mini-with-macbook-controller)
- [Configuration](#configuration)
  - [Model Configuration](#model-configuration)
  - [Cluster Configuration](#cluster-configuration)
- [Deployment](#deployment)
  - [Standard Deployment (Mac Mini Cluster)](#standard-deployment-mac-mini-cluster)
  - [Hybrid Deployment (iPad + Mac Mini)](#hybrid-deployment-ipad--mac-mini)
- [API Usage](#api-usage)
  - [Health Check](#health-check)
  - [Text Generation](#text-generation)
  - [Stream Generation](#stream-generation)

---

## Overview

Model Parallelism enables large language model inference by splitting model layers across multiple workers, allowing inference on models that exceed a single device's memory capacity. The system uses a leader-worker architecture where activations are sequentially forwarded through distributed transformer layers.

## Supported Deployment Configurations

### Configuration 1: Mac Mini Cluster (CPU-based)
- **Server (Rank 0)**: Mac mini - Layers 0-4
- **Worker 1 (Rank 1)**: Mac mini - Layers 5-9
- **Worker 2 (Rank 2)**: Raspberry Pi / Mac mini - Layers 10-11
- **Client**: FastAPI backend + Web/React frontend

### Configuration 2: iPad + Mac Mini Hybrid Cluster (CoreML + CPU)
- **Server (Rank 0)**: Mac mini M4 - Layers 0-3 (CPU/PyTorch)
- **Worker 1 (Rank 1)**: iPad - Layers 4-7 (CoreML accelerated)
- **Worker 2 (Rank 2)**: Mac mini M4 - Layers 8-11 (CPU/PyTorch)
- **Controller**: MacBook - FastAPI backend + Web frontend
- **Benefits**: Leverages iPad's Neural Engine for middle layers, offloading computation from Mac minis

## Architecture

### Standard Configuration (Mac Mini Cluster)

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

### Hybrid Configuration (iPad + Mac Mini with MacBook Controller)

![iPad Hybrid Architecture](../images/ipad_arch.png)

**Data Flow:**
1. **User Input**: User types prompt in React frontend (browser on MacBook)
2. **API Processing**: FastAPI backend on MacBook receives prompt via HTTP POST
3. **Cluster Connection**: API establishes socket connection to server (Mac mini M4 Rank 0)
4. **Tokenization**: Server tokenizes input and processes through layers 0-3
5. **Activation Forwarding**: Activations sent to iPad (Rank 1) via network
6. **CoreML Processing**: iPad runs layers 4-7 using Neural Engine acceleration
7. **Final Layers**: iPad forwards to Mac mini M4 (Rank 2) for layers 8-11
8. **Token Generation**: Rank 2 generates next token, sends back through chain
9. **Streaming Response**: API streams generated tokens back to frontend in real-time

**Components:**
- **MacBook Controller**: Hosts FastAPI backend and React frontend, manages user interaction
- **Server (Mac mini M4, Rank 0)**: Entry point, tokenization, layers 0-3, coordinates workers
- **Worker 1 (iPad, Rank 1)**: CoreML-accelerated middle layers 4-7, Swift app handles networking
- **Worker 2 (Mac mini M4, Rank 2)**: Final layers 8-11, language model head, token sampling
- **Network**: Thunderbolt between Mac minis, WiFi for iPad, all on same subnet

## Configuration

### Model Configuration

Edit `src/smolcluster/configs/model_parallelism/model_config_inference.yaml`:

**Standard Configuration (3 Mac minis):**
```yaml
causal_gpt2:
  hf_model_name: "gpt2"
  weights_model_name: "gpt2"  # Auto-downloads from HuggingFace
  num_layers: 12
  num_nodes: 3  # Server + 2 workers
  max_new_tokens: 256
  
  # Decoding strategies
  active_decoding_strategy: "top_k"
  decoding_strategies:
    top_k:
      temperature: 1.0
      k: 40
```

**Hybrid Configuration (iPad + 2 Mac mini M4):**
```yaml
causal_gpt2:
  hf_model_name: "gpt2"
  weights_model_name: "gpt2"
  num_layers: 12
  num_nodes: 3  # Mac mini (server) + iPad (worker 1) + Mac mini (worker 2)
  max_new_tokens: 256
  
  # Layer distribution for hybrid setup
  layer_distribution:
    rank_0: [0, 1, 2, 3]        # Mac mini M4 - First 4 layers
    rank_1: [4, 5, 6, 7]        # iPad CoreML - Middle 4 layers
    rank_2: [8, 9, 10, 11]      # Mac mini M4 - Last 4 layers
  
  # CoreML-specific settings for iPad
  use_coreml_rank1: true        # Enable CoreML for worker 1 (iPad)
  
  active_decoding_strategy: "top_k"
  decoding_strategies:
    top_k:
      temperature: 1.0
      k: 40
```

**Key Parameters:**
- `num_nodes`: Total nodes (server + workers)
- `num_layers`: Total transformer layers to distribute (GPT-2 has 12)
- `layer_distribution`: Custom layer assignment per rank (for hybrid setups)
- `use_coreml_rank1`: Enable CoreML acceleration on iPad worker
- `weights_model_name`: Model identifier for auto-downloading safetensors
- `active_decoding_strategy`: Default strategy (overridable via API)

### Cluster Configuration

Edit `src/smolcluster/configs/model_parallelism/cluster_config_inference.yaml`:

**Standard Configuration:**
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

**Hybrid Configuration (iPad + Mac mini with MacBook controller):**
```yaml
host_ip:
  mini1: "10.10.0.1"     # Server (Mac mini M4 - Rank 0)
  ipad: "10.10.0.5"      # Worker 1 (iPad - Rank 1, WiFi/Thunderbolt network)
  mini2: "10.10.0.2"     # Worker 2 (Mac mini M4 - Rank 2)
  macbook: "10.10.0.3"   # Controller (MacBook - FastAPI backend)

port: 65432
num_workers: 2
workers: [ipad, mini2]
server: mini1

# Web interface configuration
web_interface:
  api_host: "10.10.0.3"  # MacBook IP
  api_port: 8000         # FastAPI backend port
  frontend_port: 3000    # React frontend port
```

### CoreML Model Conversion for iPad

To enable CoreML acceleration on iPad, you need to convert the GPT-2 layers to CoreML format:

**1. Export PyTorch layers to CoreML (run on Mac):**

```bash
cd src/smolcluster/utils
python convert_to_coreml.py --model gpt2 --rank 1 --layers 4,5,6,7
```

This generates `gpt2_rank1.mlpackage` containing layers 4-7 optimized for iPad's Neural Engine.

**2. Copy CoreML model to iPad:**

```bash
# Transfer via AirDrop or copy to iPad app bundle
cp src/data/coremlmodel/gpt2_rank1.mlpackage ~/iPad/GPT2Node/Models/
```

**3. iPad Swift App Configuration:**

The iPad runs a Swift app (`ios/frontend/swift_app/GPT2Node/`) that:
- Loads CoreML model on startup
- Listens for socket connections from server
- Receives activations, runs inference, forwards to next rank
- Optimizes Neural Engine usage for low latency

**CoreML Model Structure:**
```
gpt2_rank1.mlpackage/
├── Data/
│   └── weights/          # Quantized weights for layers 4-7
├── Manifest.json
└── Metadata/
    └── com.apple.CoreML/
        └── model.mlmodel  # Neural Network architecture
```

### Model Weights

Weights are automatically downloaded on first run from HuggingFace. Supported models are in `src/smolcluster/configs/model_weights.yaml`:


**Storage:** Downloaded to `src/data/<filename>`


**Single-Script Deployment:**

For convenience, use the provided launch script to start the entire distributed inference stack with one command:

```bash
cd ~/Desktop/smolcluster/scripts/inference
./launch_mp_inference.sh
```

This script automatically:
1. Starts the inference server on Rank 0 (Mac mini)
2. SSH into workers and starts worker processes on Rank 1, 2, ... (iPad, Mac mini)
3. Launches FastAPI backend on controller (MacBook)
4. Starts frontend server on `http://localhost:3000`
5. Configures all network connections and layer distributions



**API Endpoints:**
- `POST /chat`: Submit inference requests
- `GET /config`: Retrieve active model configuration
- `GET /health`: Health check
- `POST /reconnect`: Reconnect to inference server

**Frontend Features:**
- Real-time chat interface
- Parameter display (tokens, temperature, top-p, top-k, strategy)
- Message history
- Auto-reconnect on disconnect

## References

- [Model Parallelism Training](./setup_cluster.md) - Distributed training guide
- [Logging Setup](./logging.md) - Centralized logging configuration
- [HuggingFace Models](https://huggingface.co/models) - Available pretrained models
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Original architecture
