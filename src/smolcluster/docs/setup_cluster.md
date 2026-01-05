
# Mac Mini Cluster – Thunderbolt + SSH Setup

This guide explains how to set up a **Mac mini cluster** for distributed training controlled via a **MacBook**, using **Thunderbolt for high-speed interconnect** and **Wi-Fi for control**.

## Hardware Topology

```
MacBook (controller)
   |
   | Thunderbolt (optional, control)  
   |
Mac mini 1 (master) —— Mac mini 2 —— Mac mini 3
      | Thunderbolt Bridge network
```

* **MacBook**: SSH controller, optional hotspot for Raspberry Pi or remote access
* **Mac mini 1**: Master node
* **Mac mini 2/3**: Worker nodes
* **Thunderbolt Bridge**: high-speed, low-latency interconnect between Mac minis

## Thunderbolt Setup (Mac minis)

1. Connect Mac minis via **Thunderbolt cables** in daisy chain or star topology.
2. On each Mac mini: **System Settings → Network → Add → Thunderbolt Bridge**
3. Assign **static IPs** for the Thunderbolt subnet (e.g., `172.31.0.x`):

| Mac mini | Thunderbolt IP |
| -------- | -------------- |
| mini1    | 172.31.0.1     |
| mini2    | 172.31.0.2     |
| mini3    | 172.31.0.3     |

## SSH Setup (Control from MacBook)

### Generate SSH key (once)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/macmini_cluster -C "macmini-cluster"
```

### Copy keys to Mac minis

```bash
ssh-copy-id -i ~/.ssh/macmini_cluster.pub yuvrajsingh1@<mini_wifi_ip>
```

### SSH config (optional but recommended)

```bash
nano ~/.ssh/config
```

```ssh
Host mini1
    HostName <mini1_wifi_ip>
    User yuvrajsingh1
    IdentityFile ~/.ssh/macmini_cluster
    IdentitiesOnly yes

Host mini2
    HostName <mini2_wifi_ip>
    User yuvrajsingh1
    IdentityFile ~/.ssh/macmini_cluster
    IdentitiesOnly yes

Host mini3
    HostName <mini3_wifi_ip>
    User yuvrajsingh1
    IdentityFile ~/.ssh/macmini_cluster
    IdentitiesOnly yes
```

## Persistent SSH sessions

Use `tmux` to prevent session drops:

```bash
ssh mini1
tmux new -s train
python train.py --rank 0
# Detach: Ctrl+B then D
```

## Launch Training Commands (SSH)

### Automated Launch Script

Use the provided `launch.sh` script in the project root for automatic deployment:

```bash
./launch.sh
```

**Test first with dry-run:**
```bash
./launch.sh --dry-run
```

This script:
- Reads `num_workers` from `src/smolcluster/configs/cluster_config.yaml`
- Launches server on `mini1` using SimpleAllReduce/server.py
- Launches workers on `mini2`, `mini3`, etc. using SimpleAllReduce/worker.py
- Uses `tmux` for persistent background sessions
- Provides status checking commands

**Requirements:** Install `yq` for YAML parsing: `brew install yq`

### Manual Launch (Alternative)

Example parallel launch from MacBook:

```bash
ssh mini1 "tmux new -d -s train 'cd ~/smolcluster && uv run python src/smolcluster/SimpleAllReduce/server.py'"
ssh mini2 "tmux new -d -s train 'cd ~/smolcluster && uv run python src/smolcluster/SimpleAllReduce/worker.py'"
ssh mini3 "tmux new -d -s train 'cd ~/smolcluster && uv run python src/smolcluster/SimpleAllReduce/worker.py'"
```

## Networking Principles

| Traffic                             | NIC / IP                            |
| ----------------------------------- | ----------------------------------- |
| Training (Mac mini ↔ Mac mini)      | Thunderbolt Bridge IPs (172.31.0.x) |
| Control / SSH (MacBook → minis)     | Wi-Fi IPs (DHCP or static)          |

---

MIT License - see LICENSE file for details

## Acknowledgments

- Built with PyTorch for neural network training
- Uses Weights & Biases for experiment tracking
- Inspired by federated learning research