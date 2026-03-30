#!/usr/bin/env bash
# Network discovery helper for smolcluster setup.
# Run this on EACH node (controller and all workers) to find interface names
# and current IPs, then follow the printed instructions to assign static IPs.
#
# Usage: ./scripts/installations/discover_network.sh

set -euo pipefail

log()  { echo "  [discover] $*"; }
ok()   { echo "  [discover] ✓ $*"; }
warn() { echo "  [discover] ⚠ $*" >&2; }
hr()   { echo ""; echo "  ────────────────────────────────────────────────"; }

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
elif [[ -f /etc/os-release ]]; then
    OS="linux"
fi

hr
echo ""
echo "  Smolcluster Network Discovery"
echo ""
echo "  Run this script on each node to find the interface name and current IP."
echo "  Then follow the instructions below to assign a static IP."
echo ""
hr

# ─── STEP 1: Current hostname ────────────────────────────────────────────────
echo ""
echo "  STEP 1 — Hostname"
echo ""
CURRENT_HOSTNAME=$(hostname)
log "Hostname: $CURRENT_HOSTNAME"

# ─── STEP 2: Network interfaces ──────────────────────────────────────────────
hr
echo ""
echo "  STEP 2 — Network Interfaces"
echo ""

if [[ "$OS" == "mac" ]]; then
    log "Detected macOS"
    echo ""
    log "Available interfaces:"
    ifconfig | grep -E '^[a-z]' | while read -r line; do
        iface=$(echo "$line" | cut -d: -f1)
        echo "    - $iface"
    done

    echo ""
    log "Active interfaces with IP addresses:"
    ifconfig | grep -A 3 'flags=.*UP' | grep -E '(^[a-z]|inet )' | while read -r line; do
        if [[ "$line" =~ ^[a-z] ]]; then
            echo ""
            echo "    $(echo "$line" | cut -d: -f1):"
        elif [[ "$line" =~ inet\ ([0-9.]+) ]]; then
            ip="${BASH_REMATCH[1]}"
            echo "      IP: $ip"
        fi
    done

    echo ""
    log "For home-network Ethernet, look for 'en0', 'en1', or a USB Ethernet adapter name."
    log "For Thunderbolt (Mac Mini direct-cable), look for 'bridge0' or 'bridge1'."

elif [[ "$OS" == "linux" ]]; then
    log "Detected Linux"
    echo ""
    log "Available interfaces:"
    ip -o link show | awk '{print "    - " $2}' | sed 's/:$//'

    echo ""
    log "Active interfaces with IP addresses:"
    ip -4 -o addr show | awk '{
        gsub(/:/, "", $2);
        printf "    %s: %s\n", $2, $4
    }'

    echo ""
    log "For cluster networking use the Ethernet interface (eth0, enp*, eno*, enP*, etc.)."
    log "Avoid: lo (loopback), docker0, veth* (virtual)."

    echo ""
    if command -v nmcli &>/dev/null; then
        log "NetworkManager connections:"
        nmcli -t -f NAME,DEVICE con show | awk -F: '{
            if ($2 != "") printf "    %-30s  device: %s\n", $1, $2
        }'
        ok "nmcli is available — use the connection NAME (not device name) in the commands below"
    else
        warn "nmcli not found. Install it: sudo apt install network-manager -y"
    fi
else
    warn "Unknown OS. Check network configuration manually."
fi

# ─── STEP 3: Current IP addressing ───────────────────────────────────────────
hr
echo ""
echo "  STEP 3 — Current IP"
echo ""

if [[ "$OS" == "mac" ]]; then
    CURRENT_IP=$(ifconfig | grep 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}')
elif [[ "$OS" == "linux" ]]; then
    CURRENT_IP=$(ip -4 addr show | grep inet | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d/ -f1)
else
    CURRENT_IP="unknown"
fi

if [[ -n "$CURRENT_IP" && "$CURRENT_IP" != "unknown" ]]; then
    log "Current IP (DHCP): $CURRENT_IP"
else
    warn "Could not determine current IP"
fi

# ─── STEP 4: Static IP assignment instructions ───────────────────────────────
hr
echo ""
echo "  STEP 4 — Assign a Static IP"
echo ""
echo "  Choose a private IP that won't conflict with your router's DHCP range."
echo "  If your router uses 192.168.1.x or 192.168.0.x, use a different third octet,"
echo "  e.g. 192.168.50.x  (controller = .100, workers = .101, .102, ...)"
echo ""

if [[ "$OS" == "linux" ]]; then
    echo "  Linux/Jetson — use nmcli:"
    echo ""
    echo "    # 1. Find your connection name (from STEP 2 above, e.g. 'Wired connection 1')"
    echo "    nmcli con show"
    echo ""
    echo "    # 2. Assign static IP (replace values in < > with yours)"
    echo "    sudo nmcli con mod \"<CONNECTION_NAME>\" \\"
    echo "      ipv4.addresses <192.168.50.101>/24 \\"
    echo "      ipv4.method manual"
    echo ""
    echo "    # 3. Apply"
    echo "    sudo nmcli con up \"<CONNECTION_NAME>\""
    echo ""
    echo "    # 4. Verify"
    echo "    ip addr show <INTERFACE>"
    echo ""
    echo "  The setting persists across reboots (saved in /etc/NetworkManager/system-connections/)."

elif [[ "$OS" == "mac" ]]; then
    echo "  macOS — use System Settings:"
    echo ""
    echo "    1. System Settings → Network"
    echo "    2. Click your Ethernet (or Thunderbolt Bridge) interface"
    echo "    3. Click Details"
    echo "    4. Configure IPv4: Manually"
    echo "    5. IP Address:   192.168.50.100  (change last octet per node)"
    echo "       Subnet Mask:  255.255.255.0"
    echo "       Router:       (leave blank for cluster-only fabric)"
    echo "    6. OK → Apply"
    echo ""
    echo "  Verify:"
    echo "    ifconfig en0   # or bridge0 for Thunderbolt"
fi

# ─── STEP 5: Fill nodes.yaml inventory ───────────────────────────────────────
hr
echo ""
echo "  STEP 5 — Fill Inventory (on controller only)"
echo ""
echo "  After assigning static IPs on ALL nodes, fill in the inventory on the controller:"
echo ""
echo "    cp scripts/installations/nodes.yaml.example ~/.config/smolcluster/nodes.yaml"
echo "    \${EDITOR:-nano} ~/.config/smolcluster/nodes.yaml"
echo ""
echo "  List only WORKER nodes (not this controller). Example:"
echo ""
echo "    nodes:"
echo "      - alias: jetson1"
echo "        ip: 192.168.50.101"
echo "        user: nvidia"
echo "      - alias: node2"
echo "        ip: 192.168.50.102"
echo "        user: youruser"
echo ""

# ─── NEXT STEPS ──────────────────────────────────────────────────────────────
hr
echo ""
echo "  NEXT STEPS"
echo ""
echo "  1. Verify ping from controller to each worker:"
echo "       ping -c 4 <worker-ip>"
echo ""
echo "  2. Verify password SSH from controller to each worker:"
echo "       ssh <user>@<worker-ip>"
echo ""
echo "  3. Fill ~/.config/smolcluster/nodes.yaml (STEP 5 above)"
echo ""
echo "  4. Run the automation scripts from the controller:"
echo "       ./scripts/installations/setup_ssh.sh"
echo "       ./scripts/installations/setup.sh"
echo ""
log "See docs/setup_network.md for troubleshooting."
echo ""
