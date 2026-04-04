#!/bin/bash

# Launch ClassicDP inference workers (rank 0 is leader), then chat API/frontend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=5"
RSYNC_RSH="ssh $SSH_OPTS"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/logging_helpers.sh"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

export WANDB_API_KEY="$WANDB_API_TOKEN"

LOCAL_USER="$(id -un 2>/dev/null || whoami)"
LOCAL_HOSTNAME="$(hostname 2>/dev/null || true)"
LOCAL_HOSTNAME_SHORT="$(hostname -s 2>/dev/null || true)"
declare -a LOCAL_NAMES=("localhost" "$LOCAL_HOSTNAME" "$LOCAL_HOSTNAME_SHORT")
declare -a LOCAL_IPS=("127.0.0.1" "::1")
while IFS= read -r ip; do
    [[ -n "$ip" ]] && LOCAL_IPS+=("$ip")
done < <(hostname -I 2>/dev/null | tr ' ' '\n')

is_local_ssh_target() {
    local node="$1"
    local cfg_user
    local cfg_host
    local name
    local ip

    cfg_user="$(ssh -G "$node" 2>/dev/null | awk '/^user / {print $2; exit}')"
    cfg_host="$(ssh -G "$node" 2>/dev/null | awk '/^hostname / {print $2; exit}')"

    [[ -z "$cfg_user" ]] && cfg_user="$LOCAL_USER"
    [[ -z "$cfg_host" ]] && cfg_host="$node"
    [[ "$cfg_user" != "$LOCAL_USER" ]] && return 1

    for name in "${LOCAL_NAMES[@]}"; do
        [[ -n "$name" && "$cfg_host" == "$name" ]] && return 0
    done

    for ip in "${LOCAL_IPS[@]}"; do
        [[ -n "$ip" && "$cfg_host" == "$ip" ]] && return 0
    done

    return 1
}

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config not found: $CONFIG_FILE"
    exit 1
fi

# Build WORKERS array from cluster_config_inference.yaml: "hostname:rank:ip:port"
WORKERS=()
while IFS= read -r hostrank; do
    hostname=$(echo "$hostrank" | cut -d'|' -f1)
    rank=$(echo "$hostrank" | cut -d'|' -f2)
    ip=$(yq ".host_ip[\"$hostname\"] // \"\"" "$CONFIG_FILE")
    port=$(yq ".port[\"$hostname\"] // .port.default // 65432" "$CONFIG_FILE")
    WORKERS+=("${hostname}:${rank}:${ip}:${port}")
done < <(yq '.workers.regular[] | .hostname + "|" + (.rank | tostring)' "$CONFIG_FILE")

if [[ ${#WORKERS[@]} -eq 0 ]]; then
    echo "❌ No workers found in workers.regular (cluster_config_inference.yaml)"
    exit 1
fi

RANK0_ENTRY=""
for entry in "${WORKERS[@]}"; do
    rank=$(echo "$entry" | cut -d: -f2)
    if [[ "$rank" == "0" ]]; then
        RANK0_ENTRY="$entry"
        break
    fi
done

if [[ -z "$RANK0_ENTRY" ]]; then
    echo "❌ Could not find rank 0 in ClassicDP topology"
    exit 1
fi

echo "✅ Loaded ${#WORKERS[@]} workers from cluster_config_inference.yaml"

RANK0_HOST=$(echo "$RANK0_ENTRY" | cut -d: -f1)
RANK0_IP=$(echo "$RANK0_ENTRY" | cut -d: -f3)
RANK0_PORT=$(echo "$RANK0_ENTRY" | cut -d: -f4)

SSH_NODES=()
for entry in "${WORKERS[@]}"; do
    hostname=$(echo "$entry" | cut -d: -f1)
    if is_local_ssh_target "$hostname"; then
        continue
    fi
    found=false
    for node in "${SSH_NODES[@]}"; do
        if [[ "$node" == "$hostname" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == "false" ]]; then
        SSH_NODES+=("$hostname")
    fi
done

echo "🚀 Launching ClassicDP inference"
echo "📁 Project:  $PROJECT_DIR"
echo "⚙️  Config:   $CONFIG_FILE"
echo "🧠 Leader (rank 0): $RANK0_HOST @ $RANK0_IP:$RANK0_PORT"

start_logging_stack "$PROJECT_DIR"

echo "📦 Syncing code to remote ClassicDP nodes"
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        echo "   Syncing to $node..."
        rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' \
            --exclude '*.pt' --exclude '*.pth' --exclude '*.safetensors' \
            -e "$RSYNC_RSH" "$PROJECT_DIR/" "$node:$REMOTE_PROJECT_DIR/" || {
            echo "❌ Error: Failed to sync code to $node"
            exit 1
        }
        echo "   ✅ Code synced to $node"
    done
else
    for node in "${SSH_NODES[@]}"; do
        echo "   [DRY RUN] rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' --exclude '*.pt' --exclude '*.pth' --exclude '*.safetensors' $PROJECT_DIR/ $node:$REMOTE_PROJECT_DIR/"
    done
fi

echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        if ! ssh $SSH_OPTS "$node" "echo 'SSH OK'"; then
            echo "❌ Error: Cannot connect to $node via SSH. Please check SSH setup."
            exit 1
        fi

        if ! ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && which tmux >/dev/null 2>&1"; then
            echo "❌ Error: tmux is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi

        if ! ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && uv --version >/dev/null 2>&1"; then
            echo "❌ Error: uv is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi

        if ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && (promtail --version || promtail.exe --version || which promtail || where promtail.exe || test -f /c/promtail/promtail.exe || test -f /mnt/c/promtail/promtail.exe || test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" || test -f \"C:\\\\promtail\\\\promtail.exe\")" &>/dev/null; then
            echo "🧹 $node: Cleaning up any existing Promtail processes and old logs..."
            ssh $SSH_OPTS "$node" "((pkill -f '[p]romtail' 2>/dev/null || (command -v sudo >/dev/null 2>&1 && sudo -n pkill -f '[p]romtail' 2>/dev/null) || true); (taskkill /F /IM promtail.exe >/dev/null 2>&1 || true))" &>/dev/null || true
            ssh $SSH_OPTS "$node" "rm -f $REMOTE_PROJECT_DIR/logging/cluster-logs/*.log /tmp/promtail-positions.yaml /tmp/positions.yaml" &>/dev/null || true
            ssh $SSH_OPTS "$node" "mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs"
            sleep 1

            echo "🚀 $node: Starting Promtail..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && PROMTAIL_CMD=\$(command -v promtail || command -v promtail.exe || (test -f /c/promtail/promtail.exe && echo /c/promtail/promtail.exe) || (test -f /mnt/c/promtail/promtail.exe && echo /mnt/c/promtail/promtail.exe) || (test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" && echo \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\") || (test -f \"C:\\\\promtail\\\\promtail.exe\" && echo \"C:\\\\promtail\\\\promtail.exe\") || echo promtail.exe) && nohup \$PROMTAIL_CMD -config.file=\$HOME/Desktop/smolcluster/logging/promtail-worker-remote.yaml > /tmp/promtail.log 2>&1 </dev/null &" &
            sleep 2

            if ssh $SSH_OPTS "$node" "pgrep -f promtail || tasklist /FI 'IMAGENAME eq promtail.exe' 2>nul | findstr promtail"; then
                echo "✅ $node: Promtail started successfully"
            else
                echo "⚠️  $node: Promtail may not have started. Check /tmp/promtail.log on $node"
            fi
        else
            echo "⚠️  Warning: Promtail not found on $node. Centralized logging will not work."
        fi

        echo "📦 Checking venv on $node..."
        if ! ssh $SSH_OPTS "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "⚠️  Venv not found on $node. Creating with Python 3.10..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv venv --python 3.10 .venv && uv pip install -e ."
        else
            echo "✅ Venv exists on $node. Running uv sync..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv sync"
        fi

        echo "🧪 Verifying smolcluster import on $node..."
        if ! ssh $SSH_OPTS "$node" "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -c 'import smolcluster'"; then
            echo "⚠️  Import failed on $node after sync. Reinstalling editable package..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv pip install -e ."
            if ! ssh $SSH_OPTS "$node" "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -c 'import smolcluster'"; then
                echo "❌ Error: smolcluster is not importable on $node after reinstall"
                exit 1
            fi
        fi

        echo "✅ $node: SSH OK, tmux OK, uv OK, venv OK, smolcluster import OK"
    done
else
    echo "✅ SSH and requirement checks skipped (dry run)"
fi

action_ssh() {
    local node=$1
    local command=$2
    local session_name=$3
    local log_file="$REMOTE_PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        local safe_command="$command"
        safe_command=$(echo "$safe_command" | sed -E "s/WANDB_API_KEY='[^']*'/WANDB_API_KEY='***REDACTED***'/g; s/HF_TOKEN='[^']*'/HF_TOKEN='***REDACTED***'/g")
        echo "   [DRY RUN] ssh $node \"cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \\\"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\\\"\""
        return 0
    fi

    ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\""
}

action_local() {
    local command=$1
    local session_name=$2
    local host_name
    host_name="$(hostname -s 2>/dev/null || hostname)"
    local log_file="$PROJECT_DIR/logging/cluster-logs/${session_name}__${host_name}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        local safe_command="$command"
        safe_command=$(echo "$safe_command" | sed -E "s/WANDB_API_KEY='[^']*'/WANDB_API_KEY='***REDACTED***'/g; s/HF_TOKEN='[^']*'/HF_TOKEN='***REDACTED***'/g")
        echo "   [DRY RUN] tmux new -d -s $session_name \"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\""
        return 0
    fi

    mkdir -p "$PROJECT_DIR/logging/cluster-logs"
    tmux new -d -s "$session_name" "bash -c '$command 2>&1 | tee $log_file; exec bash'"
}

echo "🧹 Cleaning up old sessions"
if [[ "$DRY_RUN" != "true" ]]; then
    for entry in "${WORKERS[@]}"; do
        hostname=$(echo "$entry" | cut -d: -f1)
        rank=$(echo "$entry" | cut -d: -f2)
        if is_local_ssh_target "$hostname"; then
            tmux kill-session -t classicdp_inf_worker$rank 2>/dev/null || true
        else
            ssh $SSH_OPTS "$hostname" "tmux kill-session -t classicdp_inf_worker$rank 2>/dev/null || true"
        fi
    done
fi

echo "👷 Starting ClassicDP inference workers"
for entry in "${WORKERS[@]}"; do
    hostname=$(echo "$entry" | cut -d: -f1)
    rank=$(echo "$entry" | cut -d: -f2)

    if is_local_ssh_target "$hostname"; then
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/ClassicDP/inference/worker.py $rank $hostname"
        action_local "$WORKER_CMD" "classicdp_inf_worker$rank"
        echo "   ✅ Local worker rank $rank for $hostname"
    else
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$REMOTE_PROJECT_DIR/src':\$PYTHONPATH && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/ClassicDP/inference/worker.py $rank $hostname"
        action_ssh "$hostname" "$WORKER_CMD" "classicdp_inf_worker$rank"
        echo "   ✅ Worker rank $rank on $hostname"
    fi
done

echo "⏳ Waiting for rank 0 to initialize"
sleep 15

echo "🌐 Launching API + frontend (targeting rank 0 leader)"
if [[ "$DRY_RUN" == "true" ]]; then
    bash "$SCRIPT_DIR/../launch_api.sh" --dry-run --backend data_parallelism --algorithm classicdp --session-prefix classicdp --server-host "$RANK0_IP" --server-port "$RANK0_PORT" --no-inference
else
    bash "$SCRIPT_DIR/../launch_api.sh" --backend data_parallelism --algorithm classicdp --session-prefix classicdp --server-host "$RANK0_IP" --server-port "$RANK0_PORT" --no-inference
fi

echo "🎉 ClassicDP inference launch complete"
