# Quickstart

Get smolcluster running on a 3-node Mac Mini cluster in under 10 minutes.

This guide covers:
1. Cluster architecture overview
2. **SyncPS DP Inference** smoke test
3. **SyncPS DP Training** smoke test
4. **Interactive dashboard** for monitoring and job control
---

## Cluster Architecture

<img src="../images/architecture.png" alt="Mac Mini Cluster Architecture" width="100%">

---


## Choose Your Setup Path

Use the path that matches your hardware before continuing:

- **Mac Mini cluster (Thunderbolt direct-cable):** Start at [Initial Mac Mini Networking Setup](#initial-mac-mini-networking-setup)
- **Jetson workers (home router / Ethernet):** Start at [Jetson Orin / Orin Nano Setup](#jetson-orin--orin-nano-setup)

After completing the matching setup section, continue to:

- [Smoke Test 1 — SyncPS DP Inference](#smoke-test-1--syncps-dp-inference)
- [Smoke Test 2 — SyncPS DP Training](#smoke-test-2--syncps-dp-training)
- [Dashboard](#dashboard)

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


### 3. Fill node inventory on `mini1`

`setup_ssh.sh` reads worker details from `~/.config/nodes.yaml`. Copy the template and fill it in:

```bash

mkdir .config
cp scripts/installations/nodes.yaml.example ./.config/nodes.yaml
${EDITOR:-nano} ./.config/nodes.yaml
```

For a Thunderbolt Mac Mini cluster the file should look like:

```yaml
nodes:
  - alias: mini2
    ip: 10.10.0.2
    user: your_username
  - alias: mini3
    ip: 10.10.0.3
    user: your_username
```

> **Home-network setup?** If your nodes are connected via a router and you don't know their IPs yet,
> run `./scripts/installations/discover_network.sh` on each node first to find the interface name
> and DHCP IP, then assign static IPs following `docs/setup_network.md`.

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



---

## Jetson Orin / Orin Nano Setup

> **Supported hardware**: Jetson AGX Orin, Orin NX, Orin Nano — all running **JetPack 6** (CUDA 12.6).
> The original Jetson Nano (JetPack 4/5) is **not** supported by these scripts.
> **Tested note**: This setup has been tested on Jetson Orin Nano with the current JetPack 6.x stack only.

These steps replace the "Initial Mac Mini Networking Setup" above when one or more workers are Jetsons connected via a home router (instead of Thunderbolt direct-cable).

---

### 1. On each Jetson worker — enable SSH

SSH may already be running. Confirm and enable on boot:

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh   # should show "active (running)"
```

---

### 2. On each Jetson worker — find the Ethernet interface and assign a static IP

Run the discovery script to see the interface name and current DHCP IP:

```bash
./scripts/installations/discover_network.sh
```

Look for your Ethernet interface in the output (typically `eth0`, `enp3s0`, `enP8p1s0`, or similar). Then assign a static IP with `nmcli`:

```bash
# List connections to get the exact NAME (e.g. "Wired connection 1")
nmcli con show

# Assign static IP — replace <CONNECTION_NAME> and choose your IP
sudo nmcli con mod "<CONNECTION_NAME>" \
  ipv4.addresses 192.168.50.101/24 \
  ipv4.method manual

# Apply
sudo nmcli con up "<CONNECTION_NAME>"

# Verify
ip addr show
```

Repeat on each Jetson worker, incrementing the last octet (`.101`, `.102`, …).

The controller (Mac mini1 or whichever machine runs the launch scripts) should also have a static IP on the same subnet — assign via System Settings → Network on Mac, or `nmcli` if it's also Linux.

---

### 3. On each Jetson worker — configure passwordless sudo

`setup_jetson.sh` installs system packages and requires passwordless sudo:

```bash
sudo visudo
# Add this line at the end (replace 'your_username' with the result of whoami):
# your_username ALL=(ALL) NOPASSWD:ALL
```

---

### 4. On controller — verify connectivity

```bash
ping -c 4 192.168.50.101   # replace with your Jetson's IP
ssh nvidia@192.168.50.101  # confirm password-based SSH works
```

---

### 5. On controller — fill node inventory

```bash
cp scripts/installations/nodes.yaml.example ~/.config/nodes.yaml
${EDITOR:-nano} ~/.config/nodes.yaml
```

Example for one Mac controller + two Jetson workers:

```yaml
nodes:
  - alias: jetson1
    ip: 192.168.50.101
    user: nvidia          # run 'whoami' on the Jetson to confirm
  - alias: jetson2
    ip: 192.168.50.102
    user: nvidia
```

---

### 6. On controller — run SSH setup and cluster bootstrap

> **Caution**
> If a node is a Jetson, `./scripts/installations/setup.sh` may invoke `scripts/installations/setup_jetson.sh` on that machine as part of the automated bootstrap.
> That Jetson setup path assumes passwordless `sudo` is already configured and will install/replace Python and PyTorch-related packages.
> If you do not want that automated behavior, read [`scripts/installations/setup_jetson.sh`](../scripts/installations/setup_jetson.sh) first and run the setup steps manually instead.

```bash
# Distribute SSH keys and write ~/.ssh/config
./scripts/installations/setup_ssh.sh

# Install base deps, clone repo, create venv on each worker
./scripts/installations/setup.sh
```

---


### 7. Copy `.env` to workers

Finally, `scp` the `.env` to each worker so they can log to W&B and access Hugging Face/wandb during training/inference:

```bash
cat > .env <<'EOF'
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_huggingface_token_here
EOF
```

Or read the aliases directly from `~/.config/nodes.yaml`:

```bash
awk '/^[[:space:]]*-[[:space:]]*alias:/ {print $3}' ~/.config/nodes.yaml |
while read -r node; do
  scp .env "$node:~/Desktop/smolcluster/"
done
```

---

### 8. Update cluster config YAMLs

Open [`src/smolcluster/configs/cluster_config_syncps.yaml`](../src/smolcluster/configs/cluster_config_syncps.yaml) and replace the `host_ip` and `workers` blocks with your actual aliases and IPs:

```yaml
host_ip:
  mini1: "192.168.50.100"    # controller
  jetson1: "192.168.50.101"
  jetson2: "192.168.50.102"

port: 65432

num_workers: 2
server: mini1
workers:
  - hostname: jetson1
    rank: 1
  - hostname: jetson2
    rank: 2
```

Do the same for [`src/smolcluster/configs/inference/cluster_config_inference.yaml`](../src/smolcluster/configs/inference/cluster_config_inference.yaml).

Then proceed to the **Smoke Test** sections below.


---

## Manual Prerequisites

> Skip this section if you used the automated setup above.

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

bash scripts/installations/installation.sh
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

## Dashboard

Run the smolcluster dashboard to monitor nodes and launch/stop jobs from a web UI.

### Step 1: Start dashboard

From the repo root:

```bash
./.venv/bin/python -m smolcluster.dashboard
```

Default bind is `0.0.0.0:9090`.

### Step 2: Open dashboard

- On the same machine: `http://localhost:9090`
- On your LAN (from another device): `http://<controller-ip>:9090`

If mDNS is working on your network, the terminal may also print a `.local` URL.

### Step 3: Optional custom host/port

```bash
./.venv/bin/python -m smolcluster.dashboard --host 0.0.0.0 --port 9090
```

### Step 4: Stop dashboard

Press `Ctrl+C` in the dashboard terminal.

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
