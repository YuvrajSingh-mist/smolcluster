#!/usr/bin/env bash
# Generates an ed25519 SSH key for smolcluster and writes clean ~/.ssh/config
# entries for every cluster node. Run once on the controller (dashboard machine).
#
# Prerequisites: complete the Thunderbolt networking and static-IP steps in
#   docs/quickstart.md before running this script. The expected node layout is:
#     mini1  10.10.0.1  (this machine — controller / rank 0)
#     mini2  10.10.0.2  (worker / rank 1)
#     mini3  10.10.0.3  (worker / rank 2)
#
# Usage: ./scripts/installations/setup_ssh.sh

set -euo pipefail

KEY_NAME="smolcluster_key"
KEY_PATH="$HOME/.ssh/$KEY_NAME"
SSH_CONFIG="$HOME/.ssh/config"
NODES_CACHE="$HOME/Desktop/smolcluster/.config/nodes"

log()  { echo "  [ssh-setup] $*"; }
ok()   { echo "  [ssh-setup] ✓ $*"; }
warn() { echo "  [ssh-setup] ⚠ $*" >&2; }
hr()   { echo ""; echo "  ────────────────────────────────────────────────"; }

# ─── STEP 1: Generate ed25519 key pair ───────────────────────────────────────
hr
echo ""
echo "  STEP 1 — SSH key"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ -f "$KEY_PATH" ]]; then
    ok "Key already exists: $KEY_PATH"
else
    log "Generating ed25519 key pair..."
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "smolcluster"
    ok "Key created: $KEY_PATH"
fi
echo ""
echo "  Public key:"
echo "    $(cat "$KEY_PATH.pub")"
echo ""

# ─── STEP 2: Load cluster nodes from nodes.yaml ──────────────────────────────
hr
echo ""
echo "  STEP 2 — Cluster nodes (from ~/.config/nodes.yaml)"
echo ""

NODES_YAML="$HOME/Desktop/smolcluster/.config/nodes.yaml"
if [[ ! -f "$NODES_YAML" ]]; then
    warn "nodes.yaml not found: $NODES_YAML"
    warn "Run discover_network.sh on each node, assign static IPs, then fill:"
    warn "  cp scripts/installations/nodes.yaml.example $NODES_YAML"
    exit 1
fi

declare -a ALIASES=() IPS=() USERS=()
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*-[[:space:]]*alias:[[:space:]]*(.+)$ ]] && ALIASES+=("${BASH_REMATCH[1]}")
    [[ "$line" =~ ^[[:space:]]*ip:[[:space:]]*(.+)$                 ]] && IPS+=("${BASH_REMATCH[1]}")
    [[ "$line" =~ ^[[:space:]]*user:[[:space:]]*(.+)$               ]] && USERS+=("${BASH_REMATCH[1]}")
done < "$NODES_YAML"

if [[ ${#ALIASES[@]} -eq 0 ]]; then
    warn "No nodes found in $NODES_YAML. Check the file format."
    exit 1
fi

for (( j=0; j<${#ALIASES[@]}; j++ )); do
    log "${ALIASES[$j]}  ${IPS[$j]}  user=${USERS[$j]}"
done
echo ""

LOCAL_USER="$(id -un 2>/dev/null || whoami)"
LOCAL_HOSTNAME="$(hostname 2>/dev/null || true)"
LOCAL_HOSTNAME_SHORT="$(hostname -s 2>/dev/null || true)"
declare -a LOCAL_NAMES=("localhost" "$LOCAL_HOSTNAME" "$LOCAL_HOSTNAME_SHORT")
declare -a LOCAL_IPS=("127.0.0.1" "::1")
while IFS= read -r ip; do
    [[ -n "$ip" ]] && LOCAL_IPS+=("$ip")
done < <(hostname -I 2>/dev/null | tr ' ' '\n')

is_local_node() {
    local alias="$1"
    local ip="$2"
    local user="$3"
    local name
    local local_ip

    [[ "$user" != "$LOCAL_USER" ]] && return 1

    for name in "${LOCAL_NAMES[@]}"; do
        [[ -n "$name" && "$alias" == "$name" ]] && return 0
    done

    for local_ip in "${LOCAL_IPS[@]}"; do
        [[ -n "$local_ip" && "$ip" == "$local_ip" ]] && return 0
    done

    return 1
}

# ─── STEP 3: Write ~/.ssh/config block ───────────────────────────────────────
hr
echo ""
echo "  STEP 3 — ~/.ssh/config"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Remove any existing smolcluster block (portable awk, works on macOS + Linux)
if grep -q "# BEGIN smolcluster" "$SSH_CONFIG" 2>/dev/null; then
    TMP=$(mktemp)
    awk '
        /# BEGIN smolcluster/ { skip=1 }
        /# END smolcluster/   { skip=0; next }
        !skip
    ' "$SSH_CONFIG" > "$TMP"
    mv "$TMP" "$SSH_CONFIG"
    log "Removed previous smolcluster block"
fi

# Append fresh block
{
    echo ""
    echo "# BEGIN smolcluster — managed by setup_ssh.sh, do not edit manually"
    for (( j=0; j<${#ALIASES[@]}; j++ )); do
        echo "Host ${ALIASES[$j]}"
        echo "    HostName ${IPS[$j]}"
        echo "    User ${USERS[$j]}"
        echo "    IdentityFile $KEY_PATH"
        echo "    IdentitiesOnly yes"
        echo "    StrictHostKeyChecking no"
        echo "    ServerAliveInterval 30"
        echo ""
    done
    echo "# END smolcluster"
} >> "$SSH_CONFIG"

ok "Config written for ${#ALIASES[@]} node(s)"
log "Preview:"
awk '/# BEGIN smolcluster/,/# END smolcluster/' "$SSH_CONFIG" | sed 's/^/    /'
echo ""

# ─── STEP 4: Copy public key to every node ───────────────────────────────────
hr
echo ""
echo "  STEP 4 — Push public key to each node"
echo "  You may be prompted for each node's password one last time."
echo ""

for (( j=0; j<${#ALIASES[@]}; j++ )); do
    node="${ALIASES[$j]}"
    user="${USERS[$j]}"
    ip="${IPS[$j]}"
    if is_local_node "$node" "$ip" "$user"; then
        ok "$node: local controller node detected, skipping ssh-copy-id"
        continue
    fi
    log "→ $node ($user@$ip) ..."
    if ssh-copy-id -i "$KEY_PATH.pub" "$node" 2>/dev/null; then
        ok "$node: passwordless SSH enabled"
    else
        warn "$node: ssh-copy-id failed — copy manually:"
        warn "  cat $KEY_PATH.pub | ssh $user@$ip 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
    fi
done

# ─── STEP 5: Persist node list for setup.sh ──────────────────────────────────
mkdir -p "$(dirname "$NODES_CACHE")"
printf '%s\n' "${ALIASES[@]}" > "$NODES_CACHE"
ok "Node list saved → $NODES_CACHE"

# ─── STEP 6: Connectivity smoke test ─────────────────────────────────────────
hr
echo ""
echo "  STEP 6 — Connectivity check"
echo ""
FAIL=0
for (( j=0; j<${#ALIASES[@]}; j++ )); do
    node="${ALIASES[$j]}"
    user="${USERS[$j]}"
    ip="${IPS[$j]}"
    if is_local_node "$node" "$ip" "$user"; then
        ok "$node: local controller node detected"
        continue
    fi
    if ssh -o ConnectTimeout=6 -o BatchMode=yes "$node" "echo ok" >/dev/null 2>&1; then
        ok "$node: SSH OK"
    else
        warn "$node: unreachable — check IP/username and retry"
        FAIL=1
    fi
done

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "  SSH setup complete. All nodes reachable."
    echo "  Run next:  ./scripts/installations/setup.sh"
else
    echo "  SSH setup done with errors. Fix connectivity before running setup.sh."
fi
echo ""
