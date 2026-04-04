#!/bin/bash

# Launch SyncPS inference server + workers, then chat API/frontend.
set -e

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

if [[ -z "$HF_TOKEN" ]]; then
    echo ""
    read -s -r -p "🔑 HF_TOKEN not set — enter HuggingFace token (required for meta-llama and other gated models): " HF_TOKEN
    echo ""
    if [[ -z "$HF_TOKEN" ]]; then
        echo "⚠️  Warning: no HF_TOKEN provided — loading gated models like Llama will fail"
    fi
    export HF_TOKEN
fi

SERVER=$(yq '.server' "$CONFIG_FILE")
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")

REGULAR_WORKERS=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && REGULAR_WORKERS+=("$worker")
done < <(yq '.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null)

TABLET_WORKERS=()
while IFS= read -r tablet; do
    [[ -n "$tablet" ]] && TABLET_WORKERS+=("$tablet")
done < <(yq '.workers.tablets[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null)

ACTUAL_WORKER_COUNT=$((${#REGULAR_WORKERS[@]} + ${#TABLET_WORKERS[@]}))
if [[ "$ACTUAL_WORKER_COUNT" -ne "$NUM_WORKERS" ]]; then
    echo "❌ Error: num_workers=$NUM_WORKERS but found $ACTUAL_WORKER_COUNT workers in config"
    exit 1
fi

# Validate total_num_nodes: SyncPS has 1 dedicated server + workers, so server counts as a node.
SYNCPS_TOTAL_NUM_NODES=$(yq '.total_num_nodes' "$CONFIG_FILE" 2>/dev/null)
SYNCPS_ACTUAL_NODES=$((ACTUAL_WORKER_COUNT + 1))  # +1 for server
if [[ -n "$SYNCPS_TOTAL_NUM_NODES" && "$SYNCPS_TOTAL_NUM_NODES" != "null" ]]; then
    if [[ "$SYNCPS_ACTUAL_NODES" -ne "$SYNCPS_TOTAL_NUM_NODES" ]]; then
        echo "❌ Node count mismatch for SyncPS:"
        echo "   total_num_nodes=$SYNCPS_TOTAL_NUM_NODES (cluster_config_inference.yaml)"
        echo "   server (1) + workers ($ACTUAL_WORKER_COUNT) = $SYNCPS_ACTUAL_NODES"
        echo "   SyncPS: server + workers must equal total_num_nodes."
        exit 1
    fi
    echo "✅ Node count OK: 1 server + $ACTUAL_WORKER_COUNT workers == total_num_nodes=$SYNCPS_TOTAL_NUM_NODES"
fi

# Sync code to all remote SyncPS nodes (server + regular workers) before launch.
SSH_NODES=("$SERVER")
SERVER_IS_LOCAL=false
if is_local_ssh_target "$SERVER"; then
    SERVER_IS_LOCAL=true
    SSH_NODES=()
fi
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
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

echo "📦 Syncing code to remote SyncPS nodes"
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
    echo "✅ Code sync complete"
else
    for node in "${SSH_NODES[@]}"; do
        echo "   [DRY RUN] rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' --exclude '*.pt' --exclude '*.pth' --exclude '*.safetensors' $PROJECT_DIR/ $node:$REMOTE_PROJECT_DIR/"
    done
    echo "✅ Code sync skipped (dry run)"
fi

echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'SSH OK'"; then
            echo "❌ Error: Cannot connect to SSH alias '$node' via key-based SSH."
            echo "   Run: ./scripts/installations/setup_ssh.sh"
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

            if [[ "$node" == "$SERVER" ]]; then
                config_file="logging/promtail-server-remote.yaml"
            else
                config_file="logging/promtail-worker-remote.yaml"
            fi

            echo "🚀 $node: Starting Promtail..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && PROMTAIL_CMD=\$(command -v promtail || command -v promtail.exe || (test -f /c/promtail/promtail.exe && echo /c/promtail/promtail.exe) || (test -f /mnt/c/promtail/promtail.exe && echo /mnt/c/promtail/promtail.exe) || (test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" && echo \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\") || (test -f \"C:\\\\promtail\\\\promtail.exe\" && echo \"C:\\\\promtail\\\\promtail.exe\") || echo promtail.exe) && nohup \$PROMTAIL_CMD -config.file=\$HOME/Desktop/smolcluster/$config_file > /tmp/promtail.log 2>&1 </dev/null &" &
            sleep 2

            if ssh $SSH_OPTS "$node" "pgrep -f promtail || tasklist /FI 'IMAGENAME eq promtail.exe' 2>nul | findstr promtail"; then
                echo "✅ $node: Promtail started successfully"
            else
                echo "⚠️  $node: Promtail may not have started. Check /tmp/promtail.log on $node"
            fi
        else
            echo "⚠️  Warning: Promtail not found on $node. Centralized logging will not work."
        fi
    done

    echo "🔧 Checking local requirements on launch machine..."
    if ! command -v tmux >/dev/null 2>&1; then
        echo "❌ Error: tmux is not installed locally. Install with: bash $PROJECT_DIR/scripts/installations/installation.sh"
        exit 1
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "❌ Error: uv is not installed locally. Install with: bash $PROJECT_DIR/scripts/installations/installation.sh"
        exit 1
    fi
else
    echo "✅ SSH and requirement checks skipped (dry run)"
fi

echo "📦 Verifying Python environments"
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        echo "   Checking venv on $node..."
        if ! ssh $SSH_OPTS "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "   ⚠️  Venv not found on $node. Creating with Python 3.10..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv venv --python 3.10 .venv && uv pip install -e ."
        else
            echo "   ✅ Venv exists on $node. Running uv sync..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv sync"
        fi

        echo "   🧪 Verifying smolcluster import on $node..."
        if ! ssh $SSH_OPTS "$node" "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -c 'import smolcluster'"; then
            echo "   ⚠️  Import failed on $node after sync. Reinstalling editable package..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv pip install -e ."
            if ! ssh $SSH_OPTS "$node" "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -c 'import smolcluster'"; then
                echo "❌ Error: smolcluster is not importable on $node after reinstall"
                exit 1
            fi
        fi

        echo "   ✅ $node: SSH OK, tmux OK, uv OK, venv OK, smolcluster import OK"
    done

    echo "   Checking local venv on launch machine..."
    if [[ ! -f "$PROJECT_DIR/.venv/bin/python" ]]; then
        echo "   ⚠️  Local venv not found. Creating with Python 3.10..."
        (
            cd "$PROJECT_DIR"
            uv venv --python 3.10 .venv
            uv pip install -e .
        )
    else
        echo "   ✅ Local venv exists. Running uv sync..."
        (
            cd "$PROJECT_DIR"
            uv sync
        )
    fi

    echo "   🧪 Verifying smolcluster import locally..."
    if ! (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}" .venv/bin/python -c "import smolcluster"); then
        echo "   ⚠️  Local import failed after sync. Reinstalling editable package..."
        (
            cd "$PROJECT_DIR"
            uv pip install -e .
        )
        if ! (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}" .venv/bin/python -c "import smolcluster"); then
            echo "❌ Error: smolcluster is not importable in local .venv after reinstall"
            exit 1
        fi
    fi

    echo "   ✅ LOCAL: tmux OK, uv OK, venv OK, smolcluster import OK"
else
    for node in "${SSH_NODES[@]}"; do
        echo "   [DRY RUN] ssh $node \"cd $REMOTE_PROJECT_DIR && if [[ ! -f .venv/bin/python ]]; then uv venv --python 3.10 .venv && uv pip install -e .; else uv sync; fi\""
    done
    echo "   [DRY RUN] cd $PROJECT_DIR && if [[ ! -f .venv/bin/python ]]; then uv venv --python 3.10 .venv && uv pip install -e .; else uv sync; fi"
fi

echo "🚀 Launching SyncPS inference"
echo "📁 Project: $PROJECT_DIR"
echo "⚙️  Config: $CONFIG_FILE"
echo "🖥️  Server: $SERVER"

start_logging_stack "$PROJECT_DIR"

action_ssh() {
    local node=$1
    local command=$2
    local session_name=$3
    local log_file="$REMOTE_PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        local safe_command="$command"
        safe_command=$(echo "$safe_command" | sed -E "s/WANDB_API_KEY='[^']*'/WANDB_API_KEY='***REDACTED***'/g; s/HF_TOKEN='[^']*'/HF_TOKEN='***REDACTED***'/g")
        echo "   [DRY RUN] ssh $node \"cd $REMOTE_PROJECT_DIR && tmux kill-session -t $session_name 2>/dev/null || true; tmux new -d -s $session_name \\\"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\\\"\""
        return 0
    fi

    ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs && cd $REMOTE_PROJECT_DIR && tmux kill-session -t $session_name 2>/dev/null || true; tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\""
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
        echo "   [DRY RUN] tmux kill-session -t $session_name 2>/dev/null || true; tmux new -d -s $session_name \"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\""
        return 0
    fi

    mkdir -p "$PROJECT_DIR/logging/cluster-logs"
    tmux kill-session -t "$session_name" 2>/dev/null || true
    tmux new -d -s "$session_name" "bash -c '$command 2>&1 | tee $log_file; exec bash'"
}

echo "🧹 Cleaning up old sessions"
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        ssh $SSH_OPTS "$node" "tmux ls 2>/dev/null | cut -d: -f1 | grep '^syncps_inf_' | xargs -I{} tmux kill-session -t {} 2>/dev/null || true"
    done

    for worker_entry in "${REGULAR_WORKERS[@]}"; do
        hostname="${worker_entry%%:*}"
        rank="${worker_entry##*:}"
        if is_local_ssh_target "$hostname"; then
            tmux kill-session -t "syncps_inf_worker$rank" 2>/dev/null || true
        else
            ssh $SSH_OPTS "$hostname" "tmux kill-session -t syncps_inf_worker$rank 2>/dev/null || true"
        fi
    done
    tmux ls 2>/dev/null | cut -d: -f1 | grep '^syncps_inf_' | xargs -I{} tmux kill-session -t {} 2>/dev/null || true
    for worker_entry in "${TABLET_WORKERS[@]}"; do
        rank="${worker_entry##*:}"
        tmux kill-session -t "syncps_inf_worker$rank" 2>/dev/null || true
    done
fi

echo "🖥️  Starting SyncPS inference server on $SERVER"
if [[ "$SERVER_IS_LOCAL" == "true" ]]; then
    echo "   ℹ️  Server alias '$SERVER' resolves to local controller; launching locally"
    SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/server.py"
    action_local "$SERVER_CMD" "syncps_inf_server"
else
    SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$REMOTE_PROJECT_DIR/src':\$PYTHONPATH && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/server.py"
    action_ssh "$SERVER" "$SERVER_CMD" "syncps_inf_server"
fi

sleep 8

echo "👷 Starting SyncPS inference workers"
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    rank="${worker_entry##*:}"
    if is_local_ssh_target "$hostname"; then
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/worker.py $rank $hostname"
        action_local "$WORKER_CMD" "syncps_inf_worker$rank"
        echo "   ✅ Local worker rank $rank for $hostname"
    else
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$REMOTE_PROJECT_DIR/src':\$PYTHONPATH && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/worker.py $rank $hostname"
        action_ssh "$hostname" "$WORKER_CMD" "syncps_inf_worker$rank"
        echo "   ✅ Worker rank $rank on $hostname"
    fi
done

for worker_entry in "${TABLET_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    rank="${worker_entry##*:}"
    WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/worker.py $rank $hostname"
    action_local "$WORKER_CMD" "syncps_inf_worker$rank"
    echo "   ✅ Local worker rank $rank for $hostname"
done

echo "⏳ Waiting for inference server to initialize"
sleep 15

echo "🌐 Launching API + frontend"
if [[ "$DRY_RUN" == "true" ]]; then
    bash "$SCRIPT_DIR/../launch_api.sh" --dry-run --backend data_parallelism --algorithm syncps --session-prefix syncps --no-inference
else
    bash "$SCRIPT_DIR/../launch_api.sh" --backend data_parallelism --algorithm syncps --session-prefix syncps --no-inference
fi

echo "🎉 SyncPS inference launch complete"
