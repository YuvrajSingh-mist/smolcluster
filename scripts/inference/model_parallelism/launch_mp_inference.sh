#!/bin/bash

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/logging_helpers.sh"

# Load environment variables from .env
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Set WANDB_API_KEY for wandb compatibility
export WANDB_API_KEY="$WANDB_API_TOKEN"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"  # Adjust if your remote path is different
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=5"
RSYNC_RSH="ssh $SSH_OPTS"

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

# Read configuration from YAML
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")
SERVER=$(yq '.server' "$CONFIG_FILE")

# Read regular workers (hostname and rank) - bash 3.2 compatible
REGULAR_WORKERS=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && REGULAR_WORKERS+=("$worker")
done < <(yq '.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null)

# Read tablet workers (hostname and rank) - bash 3.2 compatible
TABLET_WORKERS=()
while IFS= read -r tablet; do
    [[ -n "$tablet" ]] && TABLET_WORKERS+=("$tablet")
done < <(yq '.workers.tablets[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null)

# Extract just hostnames for SSH operations
WORKERS=()
for worker in "${REGULAR_WORKERS[@]}"; do
    [[ -n "$worker" ]] && WORKERS+=("${worker%%:*}")
done
TABLETS=()
for tablet in "${TABLET_WORKERS[@]}"; do
    [[ -n "$tablet" ]] && TABLETS+=("${tablet%%:*}")
done

ALL_NODES=("$SERVER" "${WORKERS[@]}" "${TABLETS[@]}")

# Validate configuration
ACTUAL_WORKER_COUNT=$((${#WORKERS[@]} + ${#TABLETS[@]}))
if [[ $ACTUAL_WORKER_COUNT -ne $NUM_WORKERS ]]; then
    echo "❌ Error: num_workers ($NUM_WORKERS) does not match total workers (${#WORKERS[@]} regular + ${#TABLETS[@]} tablets = $ACTUAL_WORKER_COUNT)"
    exit 1
fi

# Check for dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🏃 Dry run mode - willow commands without executing"
fi

echo "🚀 SmolCluster Inference Launch Script - Model Parallelism Using SyncPS "
echo "📁 Project dir: $PROJECT_DIR"
echo "⚙️  Config file: $CONFIG_FILE"

# Verify the API key works by setting it as env var and testing
export WANDB_API_KEY
if WANDB_API_KEY="$WANDB_API_KEY" wandb login --relogin <<< "$WANDB_API_KEY" 2>&1 | grep -qE "(Successfully logged in|Logged in)"; then
    echo "✅ wandb authentication successful"
else
    # Try alternative: just verify the key is valid format (40 hex chars typically)
    if [[ ${#WANDB_API_KEY} -ge 32 ]]; then
        echo "✅ API key accepted (will be set as WANDB_API_KEY on all nodes)"
    else
        echo "❌ Invalid API key format. Please check your API key."
        exit 1
    fi
fi

echo "📤 This API key will be used on all remote nodes"

# Create array of nodes that need SSH (remote server + remote regular workers only, not tablets)
SSH_NODES=()
for node in "$SERVER" "${WORKERS[@]}"; do
    [[ -z "$node" ]] && continue
    if is_local_ssh_target "$node"; then
        continue
    fi
    found=false
    for existing in "${SSH_NODES[@]}"; do
        if [[ "$existing" == "$node" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == "false" ]]; then
        SSH_NODES+=("$node")
    fi
done

echo "📦 Syncing code to remote nodes"
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

# Check SSH connectivity and remote requirements
echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ ${#TABLETS[@]} -gt 0 ]]; then
    echo "ℹ️  Skipping SSH checks for tablets: ${TABLETS[*]} (will check locally)"
fi
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        if ! ssh $SSH_OPTS "$node" "echo 'SSH OK'"; then
            echo "❌ Error: Cannot connect to $node via SSH. Please check SSH setup."
            exit 1
        fi
        
        # Check if tmux is installed on remote node
        if ! ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && which tmux"; then
            echo "❌ Error: tmux is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi
        
        # Check if uv is installed on remote node
        if ! ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && uv --version"; then
            echo "❌ Error: uv is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi
        
        # Check if Promtail is installed on remote node (cross-platform)
        if ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && (promtail --version || promtail.exe --version || which promtail || where promtail.exe || test -f /c/promtail/promtail.exe || test -f /mnt/c/promtail/promtail.exe || test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" || test -f \"C:\\\\promtail\\\\promtail.exe\")" &>/dev/null; then
            # Kill any existing Promtail processes (cleanup old/broken instances)
            echo "🧹 $node: Cleaning up any existing Promtail processes and old logs..."
            ssh $SSH_OPTS "$node" "((pkill -f '[p]romtail' 2>/dev/null || (command -v sudo >/dev/null 2>&1 && sudo -n pkill -f '[p]romtail' 2>/dev/null) || true); (taskkill /F /IM promtail.exe >/dev/null 2>&1 || true))" &>/dev/null || true
            
            # Delete old log files and position files for fresh start
            ssh $SSH_OPTS "$node" "rm -f $REMOTE_PROJECT_DIR/logging/cluster-logs/*.log /tmp/promtail-positions.yaml /tmp/positions.yaml" &>/dev/null || true
            
            # Ensure log directory exists
            ssh $SSH_OPTS "$node" "mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs"
            sleep 1
            
            # Determine config file based on node type
            if [[ "$node" == "$SERVER" ]]; then
                config_file="logging/promtail-server-remote.yaml"
            else
                config_file="logging/promtail-worker-remote.yaml"
            fi
            
            # Start Promtail in background (auto-detect path)
            echo "🚀 $node: Starting Promtail..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && PROMTAIL_CMD=\$(command -v promtail || command -v promtail.exe || (test -f /c/promtail/promtail.exe && echo /c/promtail/promtail.exe) || (test -f /mnt/c/promtail/promtail.exe && echo /mnt/c/promtail/promtail.exe) || (test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" && echo \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\") || (test -f \"C:\\\\promtail\\\\promtail.exe\" && echo \"C:\\\\promtail\\\\promtail.exe\") || echo promtail.exe) && nohup \$PROMTAIL_CMD -config.file=\$HOME/Desktop/smolcluster/$config_file > /tmp/promtail.log 2>&1 </dev/null &" &
            sleep 2
            
            # Check if Promtail is running
            if ssh $SSH_OPTS "$node" "pgrep -f promtail || tasklist /FI 'IMAGENAME eq promtail.exe' 2>nul | findstr promtail"; then
                echo "✅ $node: Promtail started successfully"
            else
                echo "⚠️  $node: Promtail may not have started. Check /tmp/promtail.log on $node"
            fi
        else
            echo "⚠️  Warning: Promtail not found on $node. Centralized logging will not work."
            echo "   Install: See logging/SETUP.md (macOS/Linux/Windows supported)"
        fi
        
        # Check that venv exists and sync dependencies
        echo "📦 Checking venv on $node..."
        if ! ssh $SSH_OPTS "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "⚠️  Venv not found on $node. Creating with Python 3.9..."
            ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv venv --python 3.9.6 .venv && source .venv/bin/activate && uv pip install -e ."
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
    
    # Check local requirements for tablet workers
    if [[ ${#TABLETS[@]} -gt 0 ]]; then
        echo ""
        echo "🔧 Checking local requirements for tablet proxy workers..."
        
        # Check tmux locally
        if ! command -v tmux &>/dev/null; then
            echo "❌ Error: tmux is not installed locally. Install with: bash $PROJECT_DIR/scripts/installations/installation.sh"
            exit 1
        fi
        
        # Check uv locally
        if ! command -v uv &>/dev/null; then
            echo "❌ Error: uv is not installed locally. Install with: bash $PROJECT_DIR/scripts/installations/installation.sh"
            exit 1
        fi
        
        # Check Promtail locally
        if command -v promtail &>/dev/null; then
            echo "🧹 LOCAL: Cleaning up any existing Promtail processes and old logs..."
            (pkill -f '[p]romtail' 2>/dev/null || (command -v sudo >/dev/null 2>&1 && sudo -n pkill -f '[p]romtail' 2>/dev/null) || true)
            rm -f "$PROJECT_DIR"/logging/cluster-logs/*.log /tmp/promtail-positions.yaml /tmp/positions.yaml 2>/dev/null || true
            mkdir -p "$PROJECT_DIR"/logging/cluster-logs
            sleep 1
            
            echo "🚀 LOCAL: Starting Promtail..."
            nohup promtail -config.file="$PROJECT_DIR/logging/promtail-worker-remote.yaml" > /tmp/promtail.log 2>&1 </dev/null &
            sleep 2
            
            if pgrep -f promtail >/dev/null; then
                echo "✅ LOCAL: Promtail started successfully"
            else
                echo "⚠️  LOCAL: Promtail may not have started. Check /tmp/promtail.log"
            fi
        else
            echo "⚠️  Warning: Promtail not found locally. Centralized logging will not work for tablets."
            echo "   Install: See logging/SETUP.md"
        fi
        
        # Check venv locally
        echo "📦 Checking local venv..."
        if [[ ! -f "$PROJECT_DIR/.venv/bin/python" ]]; then
            echo "⚠️  Venv not found locally. Creating with Python 3.9..."
            cd "$PROJECT_DIR" && uv venv --python 3.9.6 .venv && uv pip install -e .
        else
            echo "✅ Venv exists locally. Running uv sync..."
            cd "$PROJECT_DIR" && uv sync
        fi

        echo "🧪 Verifying smolcluster import locally..."
        if ! (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}" .venv/bin/python -c "import smolcluster"); then
            echo "⚠️  Local import failed after sync. Reinstalling editable package..."
            (
                cd "$PROJECT_DIR"
                uv pip install -e .
            )
            if ! (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}" .venv/bin/python -c "import smolcluster"); then
                echo "❌ Error: smolcluster is not importable in local .venv after reinstall"
                exit 1
            fi
        fi
        
        echo "✅ LOCAL: tmux OK, uv OK, venv OK, smolcluster import OK"
    fi
else
    echo "✅ SSH and remote checks skipped (dry run)"
fi



echo "Server: $SERVER"
echo "Workers: ${WORKERS[*]}"
if [[ ${#TABLETS[@]} -gt 0 ]]; then
    echo "Tablets (run manually): ${TABLETS[*]}"
fi
echo "All nodes: ${ALL_NODES[*]}"

start_logging_stack "$PROJECT_DIR"

# Function to launch on a node
launch_on_node() {
    local node=$1
    local command=$2
    local session_name=$3

    echo "🔗 Launching on $node: $command"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_file="$REMOTE_PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"
        local safe_command="$command"
        safe_command=$(echo "$safe_command" | sed -E "s/WANDB_API_KEY='[^']*'/WANDB_API_KEY='***REDACTED***'/g; s/HF_TOKEN='[^']*'/HF_TOKEN='***REDACTED***'/g")
        echo "   [DRY RUN] Would execute: ssh $node \"export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \\\"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\\\"\""
        return 0
    fi

    # SSH command with tmux and logging
    log_file="$REMOTE_PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"
    ssh $SSH_OPTS "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\"" || {
        echo "❌ Failed to launch on $node"
        return 1
    }

    echo "✅ Launched $session_name on $node (logs: $log_file)"
    
    # Give tmux a moment to start
    sleep 1
    
    # Verify session exists
    if ! ssh $SSH_OPTS "$node" "tmux has-session -t $session_name "; then
        echo "⚠️  Warning: Session $session_name on $node may have exited. Check logs: ssh $node 'tail -20 $log_file'"
    fi
}

launch_local() {
    local command=$1
    local session_name=$2
    local host_name
    host_name="$(hostname -s 2>/dev/null || hostname)"
    local log_file="$PROJECT_DIR/logging/cluster-logs/${session_name}__${host_name}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        local safe_command="$command"
        safe_command=$(echo "$safe_command" | sed -E "s/WANDB_API_KEY='[^']*'/WANDB_API_KEY='***REDACTED***'/g; s/HF_TOKEN='[^']*'/HF_TOKEN='***REDACTED***'/g")
        echo "   [DRY RUN] Would execute locally: tmux new -d -s $session_name \"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\""
        return 0
    fi

    tmux new -d -s "$session_name" "bash -c '$command 2>&1 | tee $log_file; exec bash'" || {
        echo "❌ Failed to launch local session $session_name"
        return 1
    }

    echo "✅ Launched $session_name locally (logs: $log_file)"
    sleep 1
    if ! tmux has-session -t "$session_name"; then
        echo "⚠️  Warning: Local session $session_name may have exited. Check logs: tail -20 $log_file"
    fi
}


# Kill any existing sessions
echo ""
echo "🧹 Cleaning up existing sessions..."
if [[ "$DRY_RUN" != "true" ]]; then
    if is_local_ssh_target "$SERVER"; then
        tmux kill-session -t mp_inference_server 2>/dev/null || true
    else
        ssh $SSH_OPTS "$SERVER" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && tmux kill-session -t mp_inference_server  || true"
    fi
    for worker_node in "${WORKERS[@]}"; do
        # Kill any session that starts with "mp_inference_worker"
        if is_local_ssh_target "$worker_node"; then
            tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^mp_inference_worker' | xargs -I {} tmux kill-session -t {} 2>/dev/null || true
        else
            ssh $SSH_OPTS "$worker_node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && tmux list-sessions -F '#{session_name}'  | grep -E '^mp_inference_worker' | xargs -I {} tmux kill-session -t {}  || true"
        fi
    done
    echo "✅ Cleanup complete"
else
    echo "✅ Cleanup skipped (dry run)"
fi

# Launch server on $SERVER
echo ""
echo "🖥️  Launching Model Parallelism inference server on $SERVER..."
if is_local_ssh_target "$SERVER"; then
    SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/ModelParallelism/inference/server.py"
    launch_local "$SERVER_CMD" "mp_inference_server"
else
    SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$REMOTE_PROJECT_DIR/src':\$PYTHONPATH && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/ModelParallelism/inference/server.py"
    launch_on_node "$SERVER" "$SERVER_CMD" "mp_inference_server"
fi

# Wait a moment for server to start
echo "⏳ Waiting a few seconds for server to initialize..."
sleep 10

# Launch workers
echo ""
echo "👷 Launching Model Parallelism inference workers..."

# Launch tablet workers locally
if [[ ${#TABLET_WORKERS[@]} -gt 0 ]]; then
    echo "📱 Launching tablet proxy workers locally..."
    for worker_entry in "${TABLET_WORKERS[@]}"; do
        hostname="${worker_entry%%:*}"
        rank="${worker_entry##*:}"
        
        session_name="mp_tablet_proxy$rank"
        host_name="$(hostname -s 2>/dev/null || hostname)"
        log_file="$PROJECT_DIR/logging/cluster-logs/${session_name}__${host_name}.log"
        
        # Kill existing session if it exists
        tmux kill-session -t "$session_name" 2>/dev/null || true
        
        # Launch locally in tmux
        TABLET_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/ModelParallelism/inference/worker_tablets.py $rank $hostname"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "   [DRY RUN] Would execute: tmux new -d -s $session_name \"bash -c '$TABLET_CMD 2>&1 | tee $log_file; exec bash'\""
        else
            tmux new -d -s "$session_name" "bash -c '$TABLET_CMD 2>&1 | tee $log_file; exec bash'"
            echo "   ✅ Tablet proxy rank $rank for $hostname (local session: $session_name, logs: $log_file)"
        fi
    done
fi

# Launch regular workers
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    rank="${worker_entry##*:}"

    if is_local_ssh_target "$hostname"; then
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$PROJECT_DIR/src':\$PYTHONPATH && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/ModelParallelism/inference/worker.py $rank $hostname"
        launch_local "$WORKER_CMD" "mp_inference_worker$rank"
        echo "   ✅ Local rank $rank: $hostname (mp_inference_worker$rank)"
    else
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$REMOTE_PROJECT_DIR/src':\$PYTHONPATH && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/ModelParallelism/inference/worker.py $rank $hostname"
        launch_on_node "$hostname" "$WORKER_CMD" "mp_inference_worker$rank"
        echo "   ✅ Rank $rank: $hostname (mp_inference_worker$rank)"
    fi
done

echo ""
echo "🎉 Model Parallelism inference launch complete!"
echo ""
echo "📊 Check status:"
echo "   ssh $SERVER 'tmux ls'"
echo "   ssh $SERVER 'tmux attach -t mp_inference_server'"
if [[ ${#TABLET_WORKERS[@]} -gt 0 ]]; then
    echo "   Local tablets: tmux ls (look for mp_tablet_proxy*)"
fi
echo ""
echo "💬 Server will prompt for text input. Attach to server session to interact:"
echo "   ssh $SERVER -t 'tmux attach -t mp_inference_server'"

# Wait for server to fully initialize before launching API
echo ""
echo "⏳ Waiting 30 seconds for inference server to fully initialize..."
sleep 30

# Launch API and Frontend
echo ""
echo "🌐 Launching API and Frontend..."
if [[ -f "$SCRIPT_DIR/../launch_api.sh" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        bash "$SCRIPT_DIR/../launch_api.sh" --dry-run --backend model_parallelism --algorithm mp --session-prefix mp --no-inference
    else
        bash "$SCRIPT_DIR/../launch_api.sh" --backend model_parallelism --algorithm mp --session-prefix mp --no-inference
    fi
else
    echo "⚠️  Warning: launch_api.sh not found at $SCRIPT_DIR/../launch_api.sh"
    echo "   Skipping API/Frontend launch"
fi