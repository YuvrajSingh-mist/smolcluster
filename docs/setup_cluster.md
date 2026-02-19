
# SmolCluster – Hybrid Network Setup Guide

This guide explains how to set up a **hybrid distributed training cluster** using **Thunderbolt fabric** for inter-Mac communication, **Ethernet edge links** for Raspberry Pis, and **proper routing** to ensure traffic flows correctly.

## Table of Contents

- [Network Topology](#network-topology)
- [Hardware Components](#hardware-components)
- [Part 1: Thunderbolt Fabric Setup (Mac minis)](#part-1-thunderbolt-fabric-setup-mac-minis)
  - [Physical Setup](#physical-setup)
  - [IP Assignment](#ip-assignment)
  - [Verification](#verification)
- [Part 2: Ethernet Edge Links (Pis ↔ Macs)](#part-2-ethernet-edge-links-pis--macs)
  - [Mac mini 1 Ethernet (Pi 5 Link)](#mac-mini-1-ethernet-pi-5-link)
  - [Mac mini 3 Ethernet (Pi 4 Link)](#mac-mini-3-ethernet-pi-4-link)
  - [Pi 5 Network Setup](#pi-5-network-setup)
  - [Pi 4 Network Setup](#pi-4-network-setup)
  - [Key Routing Insights](#key-routing-insights)
- [Part 3: Network Verification](#part-3-network-verification)
  - [From Pi 4](#from-pi-4)
  - [From Pi 5](#from-pi-5)
  - [From Mac minis](#from-mac-minis)
- [Part 4: SSH Setup (Control from MacBook)](#part-4-ssh-setup-control-from-macbook)
  - [Generate SSH key (once)](#generate-ssh-key-once)
  - [Find WiFi IPs for SSH](#find-wifi-ips-for-ssh)
  - [Copy keys to all nodes](#copy-keys-to-all-nodes)
  - [SSH config (optional but recommended)](#ssh-config-optional-but-recommended)
  - [SSH Troubleshooting](#ssh-troubleshooting)
- [Part 5: Training Launch](#part-5-training-launch)
  - [Automated Launch (Recommended)](#automated-launch-recommended)
- [Part 6: Troubleshooting](#part-6-troubleshooting)
  - [Connection Issues](#connection-issues)
  - [Mac IP Forwarding](#mac-ip-forwarding)
  - [ARP Cache Issues](#arp-cache-issues)
  - [Network Debugging Commands](#network-debugging-commands)
- [Part 7: Network Performance Verification](#part-7-network-performance-verification)
  - [Install iperf3](#install-iperf3)
  - [Bandwidth Testing](#bandwidth-testing)

---

## Network Topology

```
┌──────────────────────────────────────────────────────┐
│         Thunderbolt Fabric (10.10.0.0/24)           │
│                                                      │
│  Mac mini 1 (SERVER)  ←─→  Mac mini 2  ←─→  Mac mini 3 │
│     10.10.0.1              10.10.0.2        10.10.0.3  │
│         │                                       │      │
└─────────┼───────────────────────────────────────┼──────┘
          │                                       │
          │ Ethernet                              │ Ethernet
          │ 192.168.50.0/24                      │ 192.168.51.0/24
          │                                       │
     ┌────▼────┐                            ┌────▼────┐
     │   Pi 5  │                            │   Pi 4  │
     │ .50.2   │                            │ .51.4   │
     └─────────┘                            └─────────┘
```

**Design Principles:**
- ✅ **One subnet per physical link** (no L2 bridging between Ethernet segments)
- ✅ **Specific routes** for cluster traffic (not stealing default routes)
- ✅ **Macs act as gateways** for Pis to reach Thunderbolt fabric
- ✅ **Server listens only on Thunderbolt** (10.10.0.1:65432)

## Hardware Components

* **Mac mini 1**: Server + Pi 5 gateway (Thunderbolt + Ethernet)
* **Mac mini 2**: Worker (Thunderbolt only)
* **Mac mini 3**: Worker + Pi 4 gateway (Thunderbolt + Ethernet)
* **Pi 4, Pi 5**: Edge workers (Ethernet + Wi-Fi for internet)
* **MacBook**: Optional worker (Wi-Fi)

## Part 1: Thunderbolt Fabric Setup (Mac minis)

### Physical Setup
1. Connect Mac minis via **Thunderbolt 4 cables** (daisy chain or star topology)
2. On each Mac mini: **System Settings → Network → Thunderbolt Bridge → Configure IPv4**

### IP Assignment

Set **Manual** configuration with these static IPs:

| Mac mini | Thunderbolt IP | Subnet Mask     | Router  |
| -------- | -------------- | --------------- | ------- |
| mini1    | 10.10.0.1      | 255.255.255.0   | (empty) |
| mini2    | 10.10.0.2      | 255.255.255.0   | (empty) |
| mini3    | 10.10.0.3      | 255.255.255.0   | (empty) |

### Verification

```bash
# From any Mac mini, ping others:
ping 10.10.0.1  # mini1
ping 10.10.0.2  # mini2
ping 10.10.0.3  # mini3
```

✅ All Macs should be reachable via Thunderbolt IPs

## Part 2: Ethernet Edge Links (Pis ↔ Macs)

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

### Pi 5 Network Setup

**Connect Ethernet cable to Mac mini 1.**

**Configure static IP with NetworkManager:**
```bash
sudo nmcli con add type ethernet ifname eth0 con-name eth-static \
  ipv4.method manual \
  ipv4.addresses 192.168.50.2/24 \
  ipv4.never-default yes

# Add route to Thunderbolt network via mini1
sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 192.168.50.1"

# Bring up connection
sudo nmcli con up eth-static
```

**Verify routing:**
```bash
ip route
# Should show:
# default via <wifi-gateway> dev wlan0     ← Internet via Wi-Fi
# 10.10.0.0/24 via 192.168.50.1 dev eth0   ← Cluster via Ethernet
# 192.168.50.0/24 dev eth0                 ← Local Ethernet
```

**Test connectivity:**
```bash
ping 192.168.50.1  # Mac mini 1 Ethernet
ping 10.10.0.1     # Mac mini 1 Thunderbolt (SERVER)
```

### Pi 4 Network Setup

**Connect Ethernet cable to Mac mini 3.**

**Configure static IP:**
```bash
sudo nmcli con add type ethernet ifname eth0 con-name eth-static \
  ipv4.method manual \
  ipv4.addresses 192.168.51.4/24 \
  ipv4.never-default yes

# Add route to Thunderbolt network via mini3
sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 192.168.51.2"

# Bring up connection
sudo nmcli con up eth-static
```

**Verify routing:**
```bash
ip route
# Should show:
# default via <wifi-gateway> dev wlan0     ← Internet via Wi-Fi
# 10.10.0.0/24 via 192.168.51.2 dev eth0   ← Cluster via Ethernet  
# 192.168.51.0/24 dev eth0                 ← Local Ethernet
```

**Test connectivity:**
```bash
ping 192.168.51.2  # Mac mini 3 Ethernet
ping 10.10.0.1     # Server via routing through mini3
```

### Key Routing Insights

🎯 **What's happening:**
- Pis keep **internet via Wi-Fi** (default route unchanged)
- Pis add **specific route** to 10.10.0.0/24 via their Ethernet gateway
- Mac gateways **forward packets** from Ethernet → Thunderbolt
- Server only listens on 10.10.0.1 (Thunderbolt)

🚫 **What we're NOT doing:**
- Making Ethernet the default gateway (would break internet)
- Bridging the two Ethernet subnets (would violate L2 design)
- Having Pis talk directly (they route via Macs)

✅ **This mirrors real cluster design:**
- Thunderbolt = InfiniBand/RoCE fabric
- Ethernet = ToR (Top-of-Rack) links
- Pis = leaf compute nodes
- Routing = explicit fabric path control

## Part 3: Network Verification

Run these tests from each node to verify correct routing:

### From Pi 4
```bash
# Local Ethernet link
ping -c 3 192.168.51.2        # \u2705 Should work (Mac mini 3 Ethernet)

# Thunderbolt fabric (via routing)
ping -c 3 10.10.0.1           # \u2705 Should work (Server)
ping -c 3 10.10.0.3           # \u2705 Should work (Mac mini 3 Thunderbolt)

# Other Ethernet subnet  
ping -c 3 192.168.50.1        # \u274c Should FAIL (good! Different L2 domain)

# Internet
ping -c 3 8.8.8.8             # \u2705 Should work (via Wi-Fi)
```

### From Pi 5
```bash
# Local Ethernet link
ping -c 3 192.168.50.1        # \u2705 Should work (Mac mini 1 Ethernet)

# Thunderbolt fabric (via routing)
ping -c 3 10.10.0.1           # \u2705 Should work (Server)
ping -c 3 10.10.0.2           # \u2705 Should work (Mac mini 2)

# Other Ethernet subnet
ping -c 3 192.168.51.2        # \u274c Should FAIL (good! Separate link)

# Internet
ping -c 3 8.8.8.8             # \u2705 Should work (via Wi-Fi)
```

### From Mac minis
```bash
# Thunderbolt fabric
ping -c 3 10.10.0.1           # \u2705 All minis can reach all Thunderbolt IPs
ping -c 3 10.10.0.2
ping -c 3 10.10.0.3

# Ethernet (only on mini1 and mini3)
# On mini1:
ping -c 3 192.168.50.2        # \u2705 Should reach Pi 5

# On mini3:
ping -c 3 192.168.51.4        # \u2705 Should reach Pi 4
```

## Part 4: SSH Setup (Control from MacBook)

### Generate SSH key (once)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/macmini_cluster -C "macmini-cluster"
```

### Find WiFi IPs for SSH

**IMPORTANT:** SSH uses **Wi-Fi IPs**, not cluster IPs (Ethernet/Thunderbolt).

**Find IP on Mac mini:**
```bash
# Get WiFi IP address
ifconfig en0 | grep "inet " | awk '{print $2}'
```

**Find IP on Raspberry Pi:**
```bash
# Get WiFi IP address
ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1

# Or check DHCP lease
hostname -I
```

**Ensure all nodes are on the same WiFi network** for SSH to work!

### Copy keys to all nodes

**To Mac minis (use WiFi IP):**
```bash
ssh-copy-id -i ~/.ssh/macmini_cluster.pub yuvrajsingh1@<mini_wifi_ip>
```

**To Pis (use WiFi IP):**
```bash
ssh-copy-id -i ~/.ssh/macmini_cluster.pub pi@<pi_wifi_ip>
```

### SSH config (optional but recommended)

```bash
nano ~/.ssh/config
```

```ssh
Host mini1
    HostName <mini1_wifi_ip>     # Find with: ifconfig en0
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

Host pi4
    HostName <pi4_wifi_ip>       # Find with: hostname -I
    User pi
    IdentityFile ~/.ssh/macmini_cluster
    IdentitiesOnly yes

Host pi5
    HostName <pi5_wifi_ip>
    User pi
    IdentityFile ~/.ssh/macmini_cluster
    IdentitiesOnly yes
```

### SSH Troubleshooting

**"Connection timed out" errors:**

1. **Verify WiFi connectivity:**
   ```bash
   # From MacBook, ping the WiFi IP
   ping <node_wifi_ip>
   ```

2. **Check nodes are on same WiFi network:**
   ```bash
   # On each node
   iwgetid -r  # Linux - shows WiFi SSID
   
   # macOS
   /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I | grep SSID
   ```

3. **Verify SSH is running on target:**
   ```bash
   # On Mac mini
   sudo systemsetup -getremotelogin
   
   # On Pi
   sudo systemctl status ssh
   ```

4. **Enable SSH if needed:**
   ```bash
   # On Mac mini
   sudo systemsetup -setremotelogin on
   
   # On Pi
   sudo systemctl enable ssh
   sudo systemctl start ssh
   ```

## Part 5: Training Launch

### Automated Launch (Recommended)

Use `launch_edp.sh` for elastic distributed training:

```bash
cd /path/to/smolcluster
bash ./launch_edp.sh
```

This script:
- Starts server on mini1 (10.10.0.1)
- Launches workers on configured nodes
- Uses tmux for persistent sessions
- Handles connection retries automatically

## Part 6: Troubleshooting

### Connection Issues

**Pi cannot reach server:**
```bash
# Verify Ethernet link
ping 192.168.51.2  # (or .50.1 for Pi 5)

# Verify routing
ip route | grep 10.10.0.0

# Should see: 10.10.0.0/24 via 192.168.51.2 dev eth0

# Test server connectivity
nc -zv 10.10.0.1 65432
```

**Fix missing route:**
```bash
# Pi 4:
sudo nmcli con mod eth-static +ipv4.routes \"10.10.0.0/24 192.168.51.2\"
sudo nmcli con up eth-static

# Pi 5:
sudo nmcli con mod eth-static +ipv4.routes \"10.10.0.0/24 192.168.50.1\"
sudo nmcli con up eth-static
```

### Mac IP Forwarding

**Verify forwarding is enabled:**
```bash
sysctl net.inet.ip.forwarding
# Should return: net.inet.ip.forwarding: 1
```

**Re-enable if disabled:**
```bash
sudo sysctl -w net.inet.ip.forwarding=1
```

### ARP Cache Issues

**Clear ARP cache on Mac:**
```bash
sudo arp -a -d  # Clear all ARP entries
```

**Warm up ARP before training:**
```bash
# From Pi, ping the Mac gateway
ping -c 5 192.168.51.2
```
### Network Debugging Commands

```bash
# Show all network interfaces and IPs
ip addr show  # Linux
ifconfig      # macOS

# Show routing table
ip route      # Linux
netstat -rn   # macOS

# Trace packet path
traceroute 10.10.0.1  # See hops to server

# Monitor live traffic
sudo tcpdump -i eth0 port 65432  # Watch training traffic
```
1
## Part 7: Network Performance Verification

### Install iperf3

**On macOS (all Macs):**
```bash
brew install iperf3
```

**On Raspberry Pis (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install iperf3 -y
```

### Bandwidth Testing

**Between Macs (Thunderbolt):**
```bash
# On mini1 (server)
iperf3 -s

# On mini2 (client)
iperf3 -c 10.10.0.1
# Should see: 20-40 Gbps (Thunderbolt 4 capability)
```

**Pi to Server (via Ethernet + routing):**
```bash
# On mini1 (server)
iperf3 -s

# On Pi 4
iperf3 -c 10.10.0.1
# Should see: 800-950 Mbps (Gigabit Ethernet)
```

### Latency Testing

```bash
# From Pi 4 to server (multi-hop)
ping -c 100 10.10.0.1 | tail -1
# Expect: avg < 2ms

# From mini2 to server (direct Thunderbolt)
ping -c 100 10.10.0.1 | tail -1
# Expect: avg < 0.5ms
```

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