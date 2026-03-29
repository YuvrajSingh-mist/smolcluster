# Quickstart

Get smolcluster running on a 3-node Mac Mini cluster in under 10 minutes.

This guide covers:
1. Cluster architecture overview
2. **SyncPS DP Inference** smoke test
3. **SyncPS DP Training** smoke test

---

## Requirements
- 3x Mac Minis (I have used M4 2025) 
- 3x Thunderbolt 4 cables (for direct node-to-node connectivity)

## Initial Mac Mini Networking Setup

Configure Thunderbolt networking on each Mac before running any smolcluster launch script.

### 1. Cable topology

Connect the machines in a chain:

- `mini1` <-> `mini2`
- `mini2` <-> `mini3`

Then confirm each node sees a Thunderbolt bridge interface in macOS:

`System Settings -> Network -> Thunderbolt Bridge`

### 2. Assign static Thunderbolt IPs

On each Mac:

1. Open `System Settings -> Network -> Thunderbolt Bridge`
2. Click `Details`
3. Set `Configure IPv4` to `Manually`
4. Set values:

- `mini1`: IP `10.10.0.1`, Subnet Mask `255.255.255.0`
- `mini2`: IP `10.10.0.2`, Subnet Mask `255.255.255.0`
- `mini3`: IP `10.10.0.3`, Subnet Mask `255.255.255.0`

Gateway can be left empty for this private fabric.


### 3. Configure SSH on `mini1` to reach workers by hostname

Add the following to `~/.ssh/config` on `mini1` so all launch scripts can use short hostnames:

```
Host mini1
    HostName 10.10.0.1
    User your_username
    IdentityFile ~/.ssh/smolcluster_key

Host mini2
    HostName 10.10.0.2
    User your_username
    IdentityFile ~/.ssh/smolcluster_key

Host mini3
    HostName 10.10.0.3
    User your_username
    IdentityFile ~/.ssh/smolcluster_key
```

Replace `your_username` with the macOS user on each node (usually the same across all three). Confirm it works:

```bash
ssh mini2 "hostname && echo ok"
ssh mini3 "hostname && echo ok"
```

### 5. Verify interface and routes

```bash
ifconfig | grep -A5 "bridge\|thunderbolt"
netstat -rn | grep 10.10.0
```

You should see the Thunderbolt/bridge interface with your assigned `10.10.0.x` address.

### 5. Connectivity test from `mini1`

```bash
ping -c 4 10.10.0.2 (or mini2)
ping -c 4 10.10.0.3 (or mini3)
```

Only continue to inference/training after ping and SSH are both stable.



## Cluster Architecture

<img src="../images/architecture.png" alt="Mac Mini Cluster Architecture" width="100%">

| Node | Hostname | IP | Role |
|------|----------|----|------|
| Mac Mini 1 | `mini1` | `10.10.0.1` | Server (rank 0) |
| Mac Mini 2 | `mini2` | `10.10.0.2` | Worker (rank 1) |
| Mac Mini 3 | `mini3` | `10.10.0.3` | Worker (rank 2) |

Nodes are connected via **Thunderbolt fabric** (40 Gbps point-to-point). Static IPs are assigned on the `10.10.0.x` subnet. In SyncPS, `mini1` acts as the parameter server: workers compute gradients on their local replica, push gradients to the server, which aggregates and broadcasts updated weights each step.

---

## Prerequisites

### 1. SSH key auth to all nodes

on `mini1`:

```bash
    ssh-keygen -t ed25519 -f "smolcluster_key"
```

Then copy the public key to `mini2` and `mini3`:

```bash
# Run from mini1 — no password prompts allowed during launch
ssh-copy-id -i ~/.ssh/smolcluster_key.pub mini2
ssh-copy-id -i ~/.ssh/smolcluster_key.pub mini3
```


### 3. Install the project

```bash
git clone https://github.com/YuvrajSingh-mist/smolcluster.git
cd smolcluster
uv sync
```

### 4. Install dependencies on every node

```bash
brew install curl 

bash scripts/installations/installations.sh
```

### 3. Create `.env` in the project root

```bash
# /Users/yuvrajsingh9886/Desktop/cluster/smolcluster/.env
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_huggingface_token_here
```

---

## Smoke Test 1 — SyncPS DP Inference

### Step 1: Verify inference cluster config

Open [`src/smolcluster/configs/inference/cluster_config_inference.yaml`](src/smolcluster/configs/inference/cluster_config_inference.yaml) and confirm the IPs match your cluster:

```yaml
host_ip:
  mini1: "10.10.0.1"
  mini2: "10.10.0.2"
  mini3: "10.10.0.3"

num_workers: 2
total_num_nodes: 3

server: mini1
workers:
  regular:
    - hostname: mini2
      rank: 1
    - hostname: mini3
      rank: 2
```

### Step 2: Dry run (validates config + SSH without launching)

```bash
./scripts/inference/launch_inference.sh --algorithm syncps --dry-run
```

### Step 3: Launch inference

```bash
./scripts/inference/launch_inference.sh --algorithm syncps
```

The launcher will:
- Validate `num_workers` and `total_num_nodes` from config
- SSH into each worker and start the worker process in a tmux session
- Start the FastAPI server on `mini1` at port `8080`

### Step 4: Smoke test the endpoint

```bash
# Health check
curl http://localhost:8080/health

# Run a generation request
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 50}'
```

### Step 5: Cleanup

```bash
./scripts/inference/launch_inference.sh --cleanup
```

---

## Smoke Test 2 — SyncPS DP Training

### Step 1: Verify training cluster config

Open [`src/smolcluster/configs/cluster_config_syncps.yaml`](src/smolcluster/configs/cluster_config_syncps.yaml) and confirm:

```yaml
host_ip:
  mini1: "10.10.0.1"
  mini2: "10.10.0.2"
  mini3: "10.10.0.3"

port: 65432

num_workers: 2
server: mini1
workers:
  - hostname: mini2
    rank: 1
  - hostname: mini3
    rank: 2
```

### Step 2: Dry run (validates config + SSH without training)

```bash
./scripts/training/launch_syncps_train_gpt.sh --dry-run
```

### Step 3: Launch training

```bash
./scripts/training/launch_syncps_train_gpt.sh
```

The launcher will:
- Parse worker IPs and ranks from `cluster_config_syncps.yaml`
- SSH into each worker node and start the worker process in a tmux session
- Start the parameter server on `mini1` in a local tmux session
- Stream logs to W&B (requires `WANDB_API_KEY` in `.env`)

### Step 4: Monitor training

```bash
# Tail the server tmux session on mini1
tmux attach -t syncps_server

# Tail a worker session (replace 1 with 2 for mini3)
tmux attach -t syncps_worker_1
```

W&B dashboard will be linked in the terminal output once training starts.

### Step 5: Resume from checkpoint

```bash
./scripts/training/launch_syncps_train_gpt.sh --resume-checkpoint
```

### Step 6: Cleanup

```bash
# Kill all SyncPS training tmux sessions
tmux kill-session -t syncps_server
tmux kill-session -t syncps_worker_1
tmux kill-session -t syncps_worker_2
```

---

## What's Next

| Topic | Doc |
|-------|-----|
| Full cluster setup (Thunderbolt, SSH, static IPs) | [docs/setup_cluster.md](docs/setup_cluster.md) |
| All training algorithms (FSDP, EDP, EP, MP) | [docs/training.md](docs/training.md) |
| Inference modes (MP, ClassicDP, SyncPS) | [docs/inference.md](docs/inference.md) |
| Network configuration | [docs/networking.md](docs/networking.md) |
| Centralized logging with Grafana + Loki | [docs/logging.md](docs/logging.md) |
| Full config reference | [docs/configuration.md](docs/configuration.md) |
| REST API reference | [docs/api.md](docs/api.md) |
