# SmolCluster Network Configuration Guide

Complete guide for setting up the hybrid networking topology for distributed training across Mac minis, Raspberry Pis, and NVIDIA Jetson devices.

## Table of Contents

- [Network Topology](#network-topology)
- [Design Principles](#design-principles)
- [Prerequisites](#prerequisites)
- [Part 1: Thunderbolt Fabric (Mac minis)](#part-1-thunderbolt-fabric-mac-minis)
- [Part 2: Ethernet Edge Links (Pis & Jetson)](#part-2-ethernet-edge-links-pis--jetson)
- [Part 3: Network Verification](#part-3-network-verification)
- [Part 4: Troubleshooting](#part-4-troubleshooting)
- [Part 5: Performance Testing](#part-5-performance-testing)

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
                                            ┌─────────┐
                                            │ Jetson  │
                                            │ .51.5   │ (Optional)
                                            └─────────┘
```

**Key Components:**
- **Thunderbolt Fabric**: High-speed interconnect (40 Gbps) for Mac mini cluster coordination
- **Ethernet Edge Links**: Individual point-to-point connections for Pis/Jetson to reach the fabric
- **Wi-Fi**: Internet access for all edge workers (Pis/Jetson) - separate from cluster traffic

## Design Principles

**One subnet per physical link**  
Each Ethernet connection gets its own subnet (no L2 bridging across segments). This prevents broadcast storms and maintains clean separation.

**Specific routes for cluster traffic**  
Edge workers use explicit routes to reach the Thunderbolt fabric, not default gateway changes. Internet access stays on Wi-Fi.

**Macs as gateways**  
Mac minis with both Thunderbolt and Ethernet act as routers, forwarding packets between edge workers and the fabric.

**Server listens only on Thunderbolt**  
The training server binds to `10.10.0.1:65432`, ensuring all traffic flows through the high-speed fabric.

**Real-world cluster analog:**
- Thunderbolt = InfiniBand/RoCE fabric (compute network)
- Ethernet = ToR (Top-of-Rack) switch uplinks
- Edge workers = Leaf compute nodes
- Routing = Explicit fabric path control

---

## Prerequisites

### Hardware Requirements

**Mac minis:**
- 2-3 Mac minis with Thunderbolt 4 ports
- Thunderbolt 4 cables (40 Gbps)
- USB-C to Ethernet adapters (for edge connections)

**Edge Workers:**
- Raspberry Pi 4/5 with Ethernet + Wi-Fi
- NVIDIA Jetson Orin Nano with Ethernet + Wi-Fi
- Cat6 Ethernet cables

### Software Prerequisites

**On all edge workers (Pi & Jetson):**

Configure passwordless sudo for automated setup scripts:

```bash
# SSH into each device
sudo visudo

# Add at the end (replace 'username' with actual username):
username ALL=(ALL) NOPASSWD:ALL

# Save and exit (Ctrl+X, then Y, then Enter)
```

**Verify:**
```bash
sudo -n true && echo "Passwordless sudo working!"
```

> **Why needed:** Automated setup scripts install system packages via SSH. Without passwordless sudo, scripts fail with "sudo: a terminal is required to read the password".

---

## Part 1: Thunderbolt Fabric (Mac minis)

### Physical Setup

1. Connect Mac minis via **Thunderbolt 4 cables**
   - Daisy chain: mini1 ←→ mini2 ←→ mini3
   - Star topology also works if using a Thunderbolt hub

2. On each Mac mini:
   - Open **System Settings → Network**
   - Select **Thunderbolt Bridge**
   - Click **Details → TCP/IP**

### IP Configuration

Configure **Manual** IPv4 with these static assignments:

| Mac mini | Thunderbolt IP | Subnet Mask     | Router  |
| -------- | -------------- | --------------- | ------- |
| mini1    | 10.10.0.1      | 255.255.255.0   | (empty) |
| mini2    | 10.10.0.2      | 255.255.255.0   | (empty) |
| mini3    | 10.10.0.3      | 255.255.255.0   | (empty) |

**Important:** Leave the "Router" field empty to avoid routing conflicts.

### Verification

From any Mac mini, test connectivity:

```bash
ping -c 3 10.10.0.1  # mini1
ping -c 3 10.10.0.2  # mini2
ping -c 3 10.10.0.3  # mini3
```

All should respond with sub-millisecond latency (typically < 0.5ms).

---

## Part 2: Ethernet Edge Links (Pis & Jetson)

### Mac mini 1: Pi 5 Gateway

**System Settings → Network → Ethernet (USB-C adapter) → Details → TCP/IP:**
- IP Address: `192.168.50.1`
- Subnet Mask: `255.255.255.0`
- Router: (leave empty)

**Enable IP forwarding:**
```bash
sudo sysctl -w net.inet.ip.forwarding=1

# Make persistent across reboots:
echo "net.inet.ip.forwarding=1" | sudo tee -a /etc/sysctl.conf
```

### Mac mini 3: Pi 4 & Jetson Gateway

**System Settings → Network → Ethernet (USB-C adapter) → Details → TCP/IP:**
- IP Address: `192.168.51.2`
- Subnet Mask: `255.255.255.0`
- Router: (leave empty)

**Enable IP forwarding:**
```bash
sudo sysctl -w net.inet.ip.forwarding=1
echo "net.inet.ip.forwarding=1" | sudo tee -a /etc/sysctl.conf
```

---

### Raspberry Pi 5 Configuration

**Connect Ethernet cable from Pi 5 to Mac mini 1.**

**Configure static IP with NetworkManager:**
```bash
sudo nmcli con add type ethernet ifname eth0 con-name eth-static \
  ipv4.method manual \
  ipv4.addresses 192.168.50.2/24 \
  ipv4.never-default yes

# Add specific route to Thunderbolt fabric via mini1
sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 192.168.50.1"

# Activate connection
sudo nmcli con up eth-static
```

**Verify routing:**
```bash
ip route
# Expected output:
# default via <wifi-gateway> dev wlan0     ← Internet via Wi-Fi
# 10.10.0.0/24 via 192.168.50.1 dev eth0   ← Cluster via Ethernet
# 192.168.50.0/24 dev eth0                 ← Local Ethernet link
```

**Test connectivity:**
```bash
ping -c 3 192.168.50.1  # Gateway (Mac mini 1 Ethernet)
ping -c 3 10.10.0.1     # Server (Mac mini 1 Thunderbolt)
ping -c 3 8.8.8.8       # Internet (via Wi-Fi)
```

---

### Raspberry Pi 4 Configuration

**Connect Ethernet cable from Pi 4 to Mac mini 3.**

**Configure static IP:**
```bash
sudo nmcli con add type ethernet ifname eth0 con-name eth-static \
  ipv4.method manual \
  ipv4.addresses 192.168.51.4/24 \
  ipv4.never-default yes

# Add route to Thunderbolt fabric via mini3
sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 192.168.51.2"

# Activate connection
sudo nmcli con up eth-static
```

**Verify routing:**
```bash
ip route
# Expected output:
# default via <wifi-gateway> dev wlan0     ← Internet via Wi-Fi
# 10.10.0.0/24 via 192.168.51.2 dev eth0   ← Cluster via Ethernet
# 192.168.51.0/24 dev eth0                 ← Local Ethernet link
```

**Test connectivity:**
```bash
ping -c 3 192.168.51.2  # Gateway (Mac mini 3 Ethernet)
ping -c 3 10.10.0.1     # Server via routing through mini3
ping -c 3 8.8.8.8       # Internet (via Wi-Fi)
```

---

### NVIDIA Jetson Orin Nano Configuration

**Connect Ethernet cable from Jetson to Mac mini 3 (shares subnet with Pi 4).**

**Configure static IP:**
```bash
sudo nmcli con add type ethernet ifname eth0 con-name eth-static \
  ipv4.method manual \
  ipv4.addresses 192.168.51.5/24 \
  ipv4.never-default yes

# Add route to Thunderbolt fabric via mini3
sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 192.168.51.2"

# Activate connection
sudo nmcli con up eth-static
```

**Alternative:** Connect to Mac mini 2 with dedicated Ethernet adapter:
```bash
# Use 192.168.52.5/24 with gateway 192.168.52.1
sudo nmcli con add type ethernet ifname eth0 con-name eth-static \
  ipv4.method manual \
  ipv4.addresses 192.168.52.5/24 \
  ipv4.never-default yes

sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 192.168.52.1"
sudo nmcli con up eth-static
```

**Verify routing:**
```bash
ip route
# Expected: similar to Pi 4 output above
```

**Test connectivity:**
```bash
ping -c 3 192.168.51.2  # Gateway (Mac mini 3)
ping -c 3 10.10.0.1     # Server
ping -c 3 8.8.8.8       # Internet
```

---

### Key Routing Insights

**What's happening:**
- Edge workers keep **internet via Wi-Fi** (default route unchanged)
- Edge workers add **specific route** to `10.10.0.0/24` via their Ethernet gateway
- Mac gateways **forward packets** from Ethernet → Thunderbolt
- Server only listens on `10.10.0.1` (Thunderbolt)

**What we're NOT doing:**
- Making Ethernet the default gateway (would break internet)
- Bridging the two Ethernet subnets (would violate L2 design)
- Having edge workers talk directly to each other (they route via Macs)

**This mirrors real cluster design:**
- Thunderbolt = InfiniBand/RoCE fabric
- Ethernet = ToR (Top-of-Rack) switch uplinks
- Edge workers = Leaf compute nodes
- Routing = Explicit fabric path control

---

## Part 3: Network Verification

### From Edge Workers (Pi 4/5, Jetson)

Expected behavior: `[PASS]` = should work, `[FAIL]` = should fail (by design).

```bash
# Test local Ethernet link
ping -c 3 <gateway-ethernet-ip>   # [PASS] e.g., 192.168.51.2

# Test server via routing
ping -c 3 10.10.0.1               # [PASS] Should route through gateway

# Test cross-subnet isolation
ping -c 3 <other-subnet-ip>       # [FAIL] Should timeout (different L2 domain)

# Test internet connectivity
ping -c 3 8.8.8.8                 # [PASS] Via Wi-Fi default route
```

**Why cross-subnet fails:**  
Pi 5 (192.168.50.2) cannot reach Pi 4 (192.168.51.4) directly because they're on different L2 segments with no bridge. This is correct behavior - they should only communicate via the server on the Thunderbolt fabric.

### From Mac minis

```bash
# Test Thunderbolt fabric
ping -c 3 10.10.0.1               # [PASS] mini1
ping -c 3 10.10.0.2               # [PASS] mini2
ping -c 3 10.10.0.3               # [PASS] mini3

# Test edge workers (only from gateway Mac)
# From mini1:
ping -c 3 192.168.50.2            # [PASS] Pi 5 (mini1 is gateway)

# From mini3:
ping -c 3 192.168.51.4            # [PASS] Pi 4 (mini3 is gateway)
ping -c 3 192.168.51.5            # [PASS] Jetson (mini3 is gateway)
```

### Quick Diagnostics

```bash
# Check routing table on edge workers
ip route | grep 10.10.0.0
# Should show: 10.10.0.0/24 via <gateway-ip> dev eth0

# Check IP forwarding on Mac gateways
sysctl net.inet.ip.forwarding
# Should return: net.inet.ip.forwarding: 1

# Trace packet path
traceroute 10.10.0.1
# Should show: edge-worker → gateway → server (2-3 hops)
```

---

## Part 4: Troubleshooting

### Edge Worker Cannot Reach Server

**Symptom:** `ping 10.10.0.1` times out or fails.

**Diagnosis:**
```bash
# 1. Check local Ethernet link
ping <gateway-ethernet-ip>
# If this fails, check physical cable and interface status

# 2. Verify route exists
ip route | grep 10.10.0.0
# Should show route via gateway

# 3. Test gateway's Thunderbolt connectivity
ssh <gateway-hostname>
ping 10.10.0.1
# If this fails, check Thunderbolt fabric setup
```

**Fixes:**
```bash
# Re-add route if missing
sudo nmcli con mod eth-static +ipv4.routes "10.10.0.0/24 <gateway-ip>"
sudo nmcli con down eth-static && sudo nmcli con up eth-static

# Restart NetworkManager if persistent issues
sudo systemctl restart NetworkManager
```

### Mac Gateway Not Forwarding

**Symptom:** Can ping gateway from edge worker, but not server.

**Diagnosis:**
```bash
# On Mac gateway, check IP forwarding
sysctl net.inet.ip.forwarding
# Should return: 1
```

**Fix:**
```bash
# Re-enable IP forwarding
sudo sysctl -w net.inet.ip.forwarding=1

# Make persistent
echo "net.inet.ip.forwarding=1" | sudo tee -a /etc/sysctl.conf
```

### Internet Broken on Edge Worker

**Symptom:** Cannot reach internet after configuring cluster network.

**Diagnosis:**
```bash
# Check default route
ip route | grep default
# Should show Wi-Fi (wlan0), NOT Ethernet
```

**Fix:**
```bash
# Ensure ipv4.never-default is set
sudo nmcli con mod eth-static ipv4.never-default yes
sudo nmcli con down eth-static && sudo nmcli con up eth-static

# Verify default route
ip route | grep default
# Should point to wlan0
```

### ARP Cache Issues

**Symptom:** Intermittent connectivity, stale MAC addresses.

**Fix:**
```bash
# On Mac gateway
sudo arp -a -d                    # Clear ARP cache

# On edge worker
ip neigh flush all                # Flush neighbor table
ping -c 5 <gateway-ip>            # Repopulate ARP
```

### Connection Refused on Training Port

**Symptom:** `nc -zv 10.10.0.1 65432` fails with "Connection refused".

**Diagnosis:**
1. Verify server is running and bound to correct interface
2. Check firewall rules on server Mac

**Fix:**
```bash
# On server Mac (mini1), check listening ports
lsof -i :65432
# Should show Python process

# Temporarily disable firewall for testing
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
```

---

## Part 5: Performance Testing

### Bandwidth Testing

**Install iperf3:**
```bash
# macOS
brew install iperf3

# Linux (Pi/Jetson)
sudo apt install iperf3 -y
```

**Test Thunderbolt fabric (Mac to Mac):**
```bash
# On server Mac (mini1)
iperf3 -s

# On client Mac (mini2 or mini3)
iperf3 -c 10.10.0.1
# Expected: 20-40 Gbps (Thunderbolt 4 theoretical max is 40 Gbps)
```

**Test Ethernet edge links (Pi/Jetson to Server):**
```bash
# On server Mac (mini1)
iperf3 -s

# On edge worker (Pi/Jetson)
iperf3 -c 10.10.0.1
# Expected: 800-950 Mbps (Gigabit Ethernet minus overhead)
```

### Latency Testing

**Thunderbolt fabric:**
```bash
# From Mac to Mac
ping -c 100 10.10.0.1 | tail -1
# Expected: avg < 0.5ms
```

**Ethernet via routing:**
```bash
# From edge worker to server
ping -c 100 10.10.0.1 | tail -1
# Expected: avg < 2ms (includes routing overhead)
```

### Packet Loss Testing

```bash
# Long-duration test
ping -i 0.2 -c 1000 10.10.0.1
# Expected: 0% packet loss

# If packet loss > 1%, check:
# - Cable quality
# - Network interface errors: ifconfig <interface>
# - System load: top
```

---

## Network Performance Expectations

| Link Type | Bandwidth | Latency | Use Case |
|-----------|-----------|---------|----------|
| Thunderbolt (Mac-Mac) | 20-40 Gbps | < 0.5ms | Gradient aggregation, parameter sync |
| Ethernet (Edge-Server) | 800-950 Mbps | 1-2ms | Edge worker gradients, model updates |
| Wi-Fi (Internet) | 50-500 Mbps | 10-50ms | Dependency downloads, W&B logging |

---

## Common Network Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ping` to gateway works, server fails | IP forwarding disabled on Mac | Enable with `sudo sysctl -w net.inet.ip.forwarding=1` |
| Internet broken after setup | Ethernet became default route | Ensure `ipv4.never-default yes` in nmcli config |
| Cannot SSH to Mac mini | Using cluster IP instead of Wi-Fi | SSH uses Wi-Fi IPs, not Thunderbolt (see SSH setup guide) |
| Thunderbolt link down | Cable loose or interface disabled | Check System Settings → Network → Thunderbolt Bridge |
| Slow training throughput | Bandwidth saturated or high latency | Run iperf3 and ping tests, check for network errors |
| Edge worker offline | Physical cable or interface issue | Check `ip link show eth0` and cable connections |

---

## Advanced Configuration

### Multiple Edge Workers per Gateway

You can connect multiple edge workers to one Mac gateway using an Ethernet switch:

```
Mac mini 3 (192.168.51.2)
    │
    ├─ Ethernet Switch (192.168.51.0/24)
          ├─ Pi 4 (192.168.51.4)
          ├─ Jetson (192.168.51.5)
          └─ Pi Zero (192.168.51.6)
```

All use the same gateway and route configuration.

### Static ARP Entries (Optional)

For deterministic performance in high-frequency communication:

```bash
# On Mac gateway
sudo arp -s 192.168.51.4 <pi4-mac-address>

# On edge worker
sudo arp -s 192.168.51.2 <gateway-mac-address>
```

