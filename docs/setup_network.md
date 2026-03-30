# Network Setup Guide

This guide walks you through configuring static IP addresses on your cluster nodes **before** running any automation scripts. You must complete these steps first, as the automation scripts (`setup_ssh.sh` and `setup.sh`) require working network connectivity to function.

---

## Overview

Setting up a smolcluster requires:

1. **Discovery** — Find each node's current network state (IP, hostname, interface names)
2. **Static IP Assignment** — Configure persistent static IPs on a private subnet
3. **Verification** — Test connectivity with `ping` and password-based SSH
4. **Inventory Creation** — Record your node details in `~/.config/smolcluster/nodes.yaml`
5. **Automation** — Run `setup_ssh.sh` and `setup.sh` to complete cluster setup

This guide covers steps 1-3. Steps 4-5 are handled by the automation scripts.

---

## Recommended Network Configuration

### Subnet Choice

Use **`192.168.50.x/24`** for your cluster network. This avoids conflicts with typical home router DHCP ranges (`192.168.0.x`, `192.168.1.x`).

Example IP assignments:
- Controller/Server: `192.168.50.100`
- Worker 1: `192.168.50.101`
- Worker 2: `192.168.50.102`
- Worker 3: `192.168.50.103`

### Connection Types

- **Mac Mini clusters**: Use **Thunderbolt Bridge** (40 Gbps point-to-point) with daisy-chained cables
- **Mixed clusters**: Use **Ethernet** connections via a switch or direct cables
- **Jetson clusters**: Use **Ethernet** (typically `eth0` or similar interface)

Both connection types work identically once static IPs are configured.

---

## Step 1: Discovery

Before assigning static IPs, identify each node's current network state.

### On Each Node (Linux/Jetson)

#### Find Current IP Address

```bash
# Show all network interfaces and IPs
ip addr show

# Or use hostname command
hostname -I
```

Look for your current IP (often `192.168.1.x` or `192.168.0.x` from DHCP).

#### Find Interface Name

```bash
# List all network interfaces
ip -o link show

# Expected output:
# 1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 ...
# 2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ...
# 3: wlan0: <BROADCAST,MULTICAST> mtu 1500 ...
```

Common interface names:
- **Ethernet**: `eth0`, `enp8s0`, `eno1`, `enx...` (USB Ethernet dongles)
- **WiFi**: `wlan0`, `wlp...`
- **Avoid**: `lo` (loopback), `docker0`, `veth...` (virtual interfaces)

Use the **Ethernet** interface for cluster networking.

#### Find Hostname

```bash
hostnamectl
```

Note the `Static hostname` value (e.g., `jetson-1`, `ubuntu-desktop`).

#### Find Current Network Manager Connections

```bash
# List all NetworkManager connections
nmcli con show

# Example output:
# NAME                UUID                                  TYPE      DEVICE
# Wired connection 1  abc-123-def                          ethernet  eth0
# WiFi Home           xyz-456-ghi                          wifi      wlan0
```

The `NAME` column shows connection names you'll use with `nmcli` commands later.

### On Each Node (Mac)

#### Find Current IP Address

```bash
ifconfig
```

Look for your active interface (often `en0` for Ethernet, `bridge0` for Thunderbolt).

#### Find Interface Name

```bash
ifconfig | grep -E '^[a-z]' | cut -d: -f1
```

Common Mac interface names:
- **Thunderbolt Bridge**: `bridge0` or `bridge1`
- **Ethernet**: `en0`, `en1`
- **WiFi**: `en0` (on MacBook), `en1`

For Mac Mini clusters with Thunderbolt, use the **Thunderbolt Bridge** interface.

#### Find Hostname

```bash
hostname
```

Or go to: **System Settings → General → Sharing** → Computer Name

### From Router/Controller (Finding Other Nodes)

If you can't physically access a node but it's on your network:

```bash
# Show all devices on local network (requires nmap)
sudo nmap -sn 192.168.1.0/24

# Or use arp cache (shows recently contacted devices)
arp -a

# Or use ip neighbor (Linux)
ip neigh show
```

Look for devices with names matching your nodes or unfamiliar MAC addresses.

---

## Step 2: Static IP Assignment

### Linux/Jetson (using nmcli)

`nmcli` (NetworkManager CLI) is the recommended tool for persistent network configuration on Jetson and Ubuntu-based systems.

#### Prerequisite: Install NetworkManager (if missing)

```bash
# Usually pre-installed on Ubuntu/Jetson
sudo apt update
sudo apt install network-manager -y
```

#### Configure Static IP

1. **List connections**:
   ```bash
   nmcli con show
   ```
   
   Note the `NAME` of your Ethernet connection (e.g., `Wired connection 1` or `eth0`).

2. **Modify connection to use static IP**:
   ```bash
   # Replace <CONNECTION_NAME> with name from step 1
   # Replace 192.168.50.101 with your chosen IP
   sudo nmcli con mod "<CONNECTION_NAME>" \
     ipv4.addresses 192.168.50.101/24 \
     ipv4.method manual
   ```

   **Important**: Include the `/24` subnet mask (not just the IP).

3. **Optionally set gateway** (only if you need internet access through another node):
   ```bash
   sudo nmcli con mod "<CONNECTION_NAME>" \
     ipv4.gateway 192.168.50.1
   ```

4. **Optionally set DNS** (for internet access):
   ```bash
   sudo nmcli con mod "<CONNECTION_NAME>" \
     ipv4.dns "8.8.8.8 8.8.4.4"
   ```

5. **Apply changes**:
   ```bash
   sudo nmcli con up "<CONNECTION_NAME>"
   ```

6. **Verify configuration**:
   ```bash
   # Check IP address
   ip addr show <INTERFACE>
   
   # Or show connection details
   nmcli con show "<CONNECTION_NAME>" | grep ipv4
   ```

   You should see:
   ```
   ipv4.method:                            manual
   ipv4.addresses:                         192.168.50.101/24
   IP4.ADDRESS[1]:                         192.168.50.101/24
   ```

#### Configuration Persists Across Reboots

NetworkManager saves configurations to `/etc/NetworkManager/system-connections/`. Your static IP will survive reboots.

#### Troubleshooting nmcli

**Problem**: Connection name has spaces or special characters

**Solution**: Wrap in quotes:
```bash
sudo nmcli con mod "Wired connection 1" ...
```

**Problem**: Changes don't apply

**Solution**: Restart NetworkManager:
```bash
sudo systemctl restart NetworkManager
sudo nmcli con up "<CONNECTION_NAME>"
```

**Problem**: IP address doesn't show up

**Solution**: Check cable is plugged in and interface is up:
```bash
ip link show <INTERFACE>
# Look for "state UP"

# If down, bring it up:
sudo ip link set <INTERFACE> up
sudo nmcli con up "<CONNECTION_NAME>"
```

---

### Mac (using System Settings GUI)

macOS uses a graphical interface for network configuration. The settings are persistent across reboots.

#### For Thunderbolt Bridge (Mac Mini Clusters)

1. **Open System Settings** → **Network**
2. Click **Thunderbolt Bridge** in the left sidebar
3. Click **Details** button
4. Set **Configure IPv4** to `Manually`
5. Enter your chosen IP:
   - **IP Address**: `192.168.50.100` (change last octet per node)
   - **Subnet Mask**: `255.255.255.0`
   - **Router**: Leave blank (no internet routing needed)
6. Click **OK**, then **Apply**

Repeat on each Mac Mini with a different IP (`.100`, `.101`, `.102`, etc.).

#### For Ethernet (Mixed Clusters)

1. **Open System Settings** → **Network**
2. Click your **Ethernet** connection (e.g., `USB 10/100/1000 LAN`)
3. Click **Details** button
4. Set **Configure IPv4** to `Manually`
5. Enter your chosen IP:
   - **IP Address**: `192.168.50.100`
   - **Subnet Mask**: `255.255.255.0`
   - **Router**: Leave blank (optional, for internet access)
6. Click **OK**, then **Apply**

#### Verify Mac Configuration

```bash
# Check IP address
ifconfig bridge0  # for Thunderbolt
# or
ifconfig en0      # for Ethernet

# Should show:
# inet 192.168.50.100 netmask 0xffffff00 broadcast 192.168.50.255
```

---

## Step 3: Verification

After configuring static IPs on all nodes, verify connectivity **before** running automation scripts.

### Test 1: Ping (from Controller to Each Worker)

```bash
# From controller node (e.g., 192.168.50.100)
ping -c 4 192.168.50.101
ping -c 4 192.168.50.102
ping -c 4 192.168.50.103
```

**Expected**: All pings succeed with `0% packet loss`.

**If ping fails**:
- Check cable is plugged in
- Verify both nodes are on the same subnet (`192.168.50.x`)
- Check firewall: `sudo ufw status` (Linux) — disable if blocking: `sudo ufw disable`
- Confirm static IP applied: `ip addr show` (Linux) or `ifconfig` (Mac)
- Try ping in reverse direction (from worker to controller)

### Test 2: SSH with Password (from Controller to Each Worker)

```bash
# Replace <user> with the actual username on the worker
# Replace <ip> with the worker's IP
ssh <user>@192.168.50.101
```

**Expected**: Password prompt appears, and you can log in.

**If SSH fails**:

**Error: "Connection refused"**
- SSH server not running on worker:
  ```bash
  # On worker:
  sudo systemctl status ssh  # Check status
  sudo systemctl start ssh   # Start if stopped
  sudo systemctl enable ssh  # Enable on boot
  ```

**Error: "No route to host"**
- Network issue — revisit ping test
- Firewall blocking port 22 — check with:
  ```bash
  sudo ufw status
  sudo ufw allow 22/tcp
  ```

**Error: "Permission denied"**
- Username incorrect — verify with `whoami` on worker
- Password incorrect — reset on worker
- SSH disabled for user — check `/etc/ssh/sshd_config` for `PermitRootLogin` and `PasswordAuthentication` settings

### Test 3: Bidirectional Connectivity

SSH from each worker back to the controller to verify full mesh connectivity:

```bash
# On worker node:
ssh <controller-user>@192.168.50.100
```

If this works, your network is ready for automation.

---

## Step 4: Create Node Inventory

Once all nodes have static IPs and SSH works, create your inventory file.

### Copy Template

```bash
# Copy example to config location
mkdir -p ~/.config/smolcluster
cp scripts/installations/nodes.yaml.example ~/.config/smolcluster/nodes.yaml
```

### Edit Inventory

Open `~/.config/smolcluster/nodes.yaml` and add your nodes:

```yaml
# Smolcluster node inventory
nodes:
  - alias: node1
    ip: 192.168.50.101
    user: yuvrajsingh
  - alias: node2
    ip: 192.168.50.102
    user: yuvrajsingh
  - alias: node3
    ip: 192.168.50.103
    user: yuvrajsingh
```

**Field descriptions**:
- `alias`: Short name for the node (used in SSH config and scripts). Use simple names like `node1`, `worker1`, `jetson1`, etc.
- `ip`: Static IP address you assigned in Step 2
- `user`: SSH username on that node (find with `whoami` on the node)

**Do NOT include** the controller/server node in this inventory — only worker nodes.

---

## Step 5: Run Automation Scripts

Now that networking is configured and verified:

1. **Generate SSH keys and distribute**:
   ```bash
   ./scripts/installations/setup_ssh.sh
   ```

   This script reads worker details from `~/.config/smolcluster/nodes.yaml` (no prompts).
   It will:
   - Generate `~/.ssh/smolcluster_key`
   - Write SSH config entries for all nodes in `nodes.yaml`
   - Copy public key to each worker (passwordless SSH)
   - Test connectivity

2. **Install dependencies and sync repository**:
   ```bash
   ./scripts/installations/setup.sh
   ```
   
   This will:
   - Install dependencies on all nodes
   - Clone/pull repository on all workers
   - Set up Python environments

See [quickstart.md](quickstart.md) for next steps after automation completes.

---

## Troubleshooting

### "SSH unreachable" from setup_ssh.sh

**Cause**: Static IP didn't persist, or firewall is blocking.

**Solution**:
- Verify IP is still configured: `ip addr show` (Linux) or `ifconfig` (Mac)
- If IP is missing, reapply with `nmcli` (Linux) or System Settings (Mac)
- Check firewall: `sudo ufw disable` (temporary, for testing)
- Verify SSH service running: `sudo systemctl status ssh`

### "Wrong interface" — IP assigned to wrong interface

**Cause**: System has multiple network interfaces, and you configured the wrong one.

**Solution**:
- Use `ip -o link show` (Linux) to list interfaces
- Find the interface matching your physical connection (eth0, bridge0, etc.)
- Delete incorrect configuration:
  ```bash
  sudo nmcli con del "<WRONG_CONNECTION>"
  ```
- Reconfigure correct interface

### "IP conflict" — address already in use

**Cause**: Another device on your network has the same IP.

**Solution**:
- Choose a different subnet that doesn't overlap with your home router's DHCP range
- If router uses `192.168.1.x`, use `192.168.50.x` for cluster
- If router uses `192.168.0.x`, use `192.168.50.x` for cluster
- Check router's DHCP range in router admin panel (usually at `192.168.1.1` or `192.168.0.1`)

### "Configuration doesn't survive reboot"

**Cause**: Using temporary methods like `sudo ip addr add` instead of `nmcli`.

**Solution**:
- **Linux**: Always use `nmcli con mod` (as shown above), NOT `ip addr add`
- **Mac**: Always use System Settings GUI, NOT `ifconfig` commands
- Verify persistence:
  ```bash
  # Reboot node
  sudo reboot
  
  # After reboot, check IP is still there:
  ip addr show <INTERFACE>  # Linux
  ifconfig <INTERFACE>      # Mac
  ```

### "nmcli: command not found" on Jetson

**Cause**: NetworkManager not installed.

**Solution**:
```bash
sudo apt update
sudo apt install network-manager -y
sudo systemctl enable NetworkManager
sudo systemctl start NetworkManager
```

### Interface is "DOWN" and won't come up

**Cause**: Cable unplugged, or driver issue.

**Solution**:
- Check physical cable connection
- Bring interface up manually:
  ```bash
  sudo ip link set <INTERFACE> up
  sudo nmcli con up "<CONNECTION_NAME>"
  ```
- Check kernel logs for driver errors:
  ```bash
  dmesg | grep -i eth
  ```

---

## Advanced: Manual Configuration (Not Recommended)

If you cannot use `nmcli` (rare), you can edit configuration files directly.

### Using netplan (Ubuntu 20.04+)

Edit `/etc/netplan/01-netcfg.yaml`:

```yaml
network:
  version: 2
  ethernets:
    eth0:
      addresses:
        - 192.168.50.101/24
      dhcp4: no
```

Apply:
```bash
sudo netplan apply
```

**Warning**: This method conflicts with NetworkManager. Use `nmcli` instead when possible.

---

## Summary Checklist

Before running automation scripts, verify:

- [ ] All nodes have static IPs on the same subnet (e.g., `192.168.50.x`)
- [ ] `ping` works from controller to all workers
- [ ] `ssh user@ip` works with password from controller to all workers
- [ ] Static IPs persist after reboot (test one node)
- [ ] `~/.config/smolcluster/nodes.yaml` created with correct aliases, IPs, and usernames

Once all checkboxes are complete, proceed to [quickstart.md](quickstart.md) → "Automated Setup" section.

