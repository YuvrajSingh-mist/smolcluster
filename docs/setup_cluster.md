
# SmolCluster – Cluster Setup Guide

Complete guide for setting up a distributed training cluster across Mac minis, Raspberry Pis, and NVIDIA Jetson devices.

## Table of Contents

- [Overview](#overview)
- [Network Setup](#network-setup)
- [Jetson GPU Worker Setup](#jetson-gpu-worker-setup)
- [SSH Configuration](#ssh-configuration)
- [Training Launch](#training-launch)
- [Network Troubleshooting](#network-troubleshooting)
- [iPad + Mac Mini Hybrid Inference](#ipad--mac-mini-hybrid-inference-cluster-setup)

---

## Overview

SmolCluster uses a **hybrid network topology**:
- **Thunderbolt fabric** (10.10.0.0/24) for high-speed Mac-to-Mac communication (20-40 Gbps)
- **Ethernet edge links** (192.168.5x.0/24) for Raspberry Pis and Jetson to reach the fabric
- **Wi-Fi** for internet access and SSH control

### Hardware Components

* **Mac mini 1**: Server + Pi 5 gateway (Thunderbolt + Ethernet)
* **Mac mini 2**: Worker (Thunderbolt only)
* **Mac mini 3**: Worker + Pi 4/Jetson gateway (Thunderbolt + Ethernet)
* **Pi 4, Pi 5**: Edge workers (Ethernet + Wi-Fi)
* **NVIDIA Jetson Orin Nano**: GPU edge worker (Ethernet + Wi-Fi)
* **MacBook**: Optional controller/worker (Wi-Fi)

---

## Network Setup

**For complete network configuration instructions, see [Network Configuration Guide](networking.md).**

The networking guide covers:
- Thunderbolt fabric configuration (Mac minis)
- Ethernet edge link setup (Pis & Jetson)
- Routing configuration and verification
- Troubleshooting common network issues
- Performance testing (bandwidth, latency)

### Quick Setup Summary

**1. Configure Thunderbolt fabric** between Mac minis:
- Assign static IPs: 10.10.0.1, 10.10.0.2, 10.10.0.3
- Leave router field empty

**2. Configure Ethernet gateways** on Mac minis:
- Mac mini 1 Ethernet: 192.168.50.1 (for Pi 5)
- Mac mini 3 Ethernet: 192.168.51.2 (for Pi 4 & Jetson)
- Enable IP forwarding: `sudo sysctl -w net.inet.ip.forwarding=1`

**3. Configure edge workers** (Pi 4, Pi 5, Jetson):
- Set static IP on Ethernet interface
- Add specific route to 10.10.0.0/24 via Mac gateway
- Keep Wi-Fi for internet access

**4. Verify connectivity:**
```bash
# From edge workers
ping 10.10.0.1  # Should reach server via routing
```

**For detailed step-by-step instructions, troubleshooting, and performance testing, see [networking.md](networking.md).**

---

## Jetson GPU Worker Setup

### Prerequisites

**Configure passwordless sudo** (required for automated setup):
```bash
sudo visudo
# Add: username ALL=(ALL) NOPASSWD:ALL
```

For network configuration, follow the [Network Configuration Guide](networking.md#nvidia-jetson-orin-nano-configuration).

### Mac mini 1 Ethernet (Pi 5 Link)

**System Settings → Network → Ethernet → Configure IPv4:**
- IP: `192.168.50.1`
- Subnet: `255.255.255.0`
- Router: (leave empty)

**Enable IP forwarding** (so Pi 5 can reach Thunderbolt network):
```bash
sudo sysctl -w net.inet.ip.forwarding=1
# Make persistent:
echo "net.inet.ip.forwarding=1" | sudo tee -a /etc/sysctl.conf
```

### Mac mini 3 Ethernet (Pi 4 Link)

**System Settings → Network → Ethernet → Configure IPv4:**
- IP: `192.168.51.2`
- Subnet: `255.255.255.0`  
- Router: (leave empty)

**Enable IP forwarding:**
```bash
sudo sysctl -w net.inet.ip.forwarding=1
echo "net.inet.ip.forwarding=1" | sudo tee -a /etc/sysctl.conf
```
### Software Setup

**Automated installation:**
```bash
cd ~/Desktop
git clone https://github.com/YuvrajSingh-mist/smolcluster.git
cd smolcluster
bash scripts/installations/setup_jetson.sh
```

The script automatically:
- Installs system dependencies (CUDA libs, OpenBLAS, OpenMPI, Python 3.10)
- Creates Python 3.10 venv with `uv`
- Installs project dependencies
- **Installs Jetson-specific PyTorch 2.8.0 with CUDA 12.6** from NVIDIA's Jetson AI Lab index
- Verifies CUDA availability

**Verify CUDA:**
```bash
source .venv/bin/activate
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
# Expected: CUDA: True, Device: Orin
```

### PyTorch Wheels Source

> **Critical**: Jetson requires custom ARM64+CUDA wheels from `https://pypi.jetson-ai-lab.io/jp6/cu126`  
> Standard `pip install torch` installs incompatible CPU-only builds. Always use the setup script.

### Performance Notes
- **GPU**: 1024 CUDA cores, 8GB shared memory
- **Power**: 7-15W (edge-optimized)
- **Bandwidth**: ~950 Mbps over Gigabit Ethernet

### Troubleshooting

| Issue | Solution |
|---|---|
| **CUDA not detected** | Run `ldconfig -p \| grep libcuda` to verify CUDA libs. Check `cat /etc/nv_tegra_release` for JetPack version. |
| **Wrong PyTorch version** | Remove `.venv` and re-run `setup_jetson.sh` to get Jetson-specific wheels. |
| **Network issues** | See [Network Configuration Guide](networking.md#troubleshooting) for routing and connectivity fixes. |

---

## SSH Configuration

### Generate SSH Key

```bash
ssh-keygen -t ed25519 -f ~/.ssh/macmini_cluster -C "macmini-cluster"
```

### Important: SSH Uses Wi-Fi IPs

**SSH connections use Wi-Fi IPs, NOT cluster IPs (Ethernet/Thunderbolt).**

**Find Wi-Fi IP on Mac mini:**
```bash
ifconfig en0 | grep "inet " | awk '{print $2}'
```

**Find Wi-Fi IP on Raspberry Pi/Jetson:**
```bash
ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
# Or: hostname -I
```

### Copy SSH Keys

```bash
# To Mac minis
ssh-copy-id -i ~/.ssh/macmini_cluster.pub yuvrajsingh1@<mini_wifi_ip>

# To Pis/Jetson
ssh-copy-id -i ~/.ssh/macmini_cluster.pub pi@<pi_wifi_ip>
```

### SSH Config (Optional)

Create `~/.ssh/config`:

```ssh
Host mini1
    HostName <mini1_wifi_ip>
    User yuvrajsingh1
    IdentityFile ~/.ssh/macmini_cluster

Host mini2
    HostName <mini2_wifi_ip>
    User yuvrajsingh1
    IdentityFile ~/.ssh/macmini_cluster

Host mini3
    HostName <mini3_wifi_ip>
    User yuvrajsingh1
    IdentityFile ~/.ssh/macmini_cluster

Host pi4
    HostName <pi4_wifi_ip>
    User pi
    IdentityFile ~/.ssh/macmini_cluster

Host pi5
    HostName <pi5_wifi_ip>
    User pi
    IdentityFile ~/.ssh/macmini_cluster

Host jetson
    HostName <jetson_wifi_ip>
    User yuvrajsingh
    IdentityFile ~/.ssh/macmini_cluster
```

### SSH Troubleshooting

**Enable SSH on nodes if needed:**
```bash
# Mac mini
sudo systemsetup -setremotelogin on

# Pi/Jetson
sudo systemctl enable ssh && sudo systemctl start ssh
```

---

## Training Launch

### Automated Launch Scripts

Launch distributed training using the provided scripts:

```bash
cd /path/to/smolcluster/scripts/training

# FSDP (ZeRO-optimized, recommended for large models)
bash launch_fsdp_train_gpt.sh

# Classic Data Parallelism
bash launch_dp_train_gpt.sh

# Elastic Distributed Parallelism
bash launch_edp_train_gpt.sh

# Model Parallelism
bash launch_mp_train_gpt.sh
```

Launch scripts automatically:
- Sync code to all nodes via rsync
- Start server on mini1 (10.10.0.1:65432)
- Launch workers on configured nodes
- Use tmux for persistent sessions
- Handle connection retries and logging

For detailed algorithm information, see [Training Guide](training.md).

---

## Network Troubleshooting

For network connectivity issues, see the comprehensive troubleshooting section in [Network Configuration Guide](networking.md#part-4-troubleshooting).

Common issues covered:
- Edge worker cannot reach server (routing problems)
- Mac gateway not forwarding packets (IP forwarding disabled)
- Internet broken on edge worker (default route issues)
- ARP cache problems
- Connection refused on training port

---

## iPad + Mac Mini Hybrid Inference Cluster Setup

This section covers the setup for distributed GPT-2 inference using **iPad (CoreML) + 2× Mac mini M4** with a MacBook controller.

### Network Topology

![iPad Hybrid Architecture](../images/ipad_arch.png)

**Hardware:**
- **Mac mini M4 #1** (Rank 0/Server): Tokenization, layers 0-3, coordination
- **iPad** (Rank 1/Worker 1): CoreML-accelerated layers 4-7
- **Mac mini M4 #2** (Rank 2/Worker 2): Layers 8-11, LM head, sampling
- **MacBook** (Controller): FastAPI backend + HTML/CSS/JS frontend

### Network Configuration

Connect Mac minis via Thunderbolt, join iPad and MacBook to same WiFi network. Configure static IPs as shown in topology diagram.

### Layer Distribution

- **Rank 0** (Mac mini M4 #1): Layers 0-3 (embedding + first 4 transformer blocks)
- **Rank 1** (iPad CoreML): Layers 4-7 (middle 4 transformer blocks, Neural Engine accelerated)
- **Rank 2** (Mac mini M4 #2): Layers 8-11 (final 4 transformer blocks + LM head)

### Setup Instructions

See [Inference Guide](inference.md) for detailed setup, CoreML conversion, and deployment instructions.

---

## Acknowledgments

- Built with PyTorch for distributed neural network training
- Uses Weights & Biases for experiment tracking
- Network topology inspired by real HPC and datacenter cluster designs
- Elastic training inspired by federated learning research
- CoreML integration for iOS/iPadOS acceleration

**MIT License** - see LICENSE file for details