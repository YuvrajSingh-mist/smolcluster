#!/bin/bash

# Load environment variables from .env
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set WANDB_API_KEY for wandb compatibility
export WANDB_API_KEY="$WANDB_API_TOKEN"

# Set CUDA environment variables (for Jetson and other CUDA devices)
if [[ -n "$CUDA_HOME" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config_edp.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"  # Adjust if your remote path is different

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/node_helpers.sh"
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/logging_helpers.sh"
init_node_helpers "$CONFIG_FILE" "$PROJECT_DIR" "$REMOTE_PROJECT_DIR"

# Read configuration from YAML
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")
SERVER=$(yq '.server' "$CONFIG_FILE")

# Read workers (hostname and rank) - bash 3.2 compatible
WORKER_ENTRIES=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && WORKER_ENTRIES+=("$worker")
done < <(yq '.workers[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE")

# Extract just hostnames for SSH operations
WORKERS=()
for worker in "${WORKER_ENTRIES[@]}"; do
    WORKERS+=("${worker%%:*}")
done

ALL_NODES=("$SERVER" "${WORKERS[@]}")

# Validate configuration
if [[ ${#WORKERS[@]} -ne $NUM_WORKERS ]]; then
    echo "❌ Error: num_workers ($NUM_WORKERS) does not match the number of workers in the list (${#WORKERS[@]})"
    exit 1
fi

# Check for dry-run flag
DRY_RUN=false
RESUME_CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            echo "🏃 Dry run mode - running commands without executing"
            shift
            ;;
        --resume-checkpoint)
            RESUME_CHECKPOINT="$2"
            echo "🔄 Will resume from checkpoint: $RESUME_CHECKPOINT"
            shift 2
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--resume-checkpoint PATH]"
            exit 1
            ;;
    esac
done

echo "🚀 SmolCluster Launch Script - EDP Version"
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

echo "📦 Syncing code to nodes..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${ALL_NODES[@]}"; do
        if node_is_local "$node"; then
            echo "   ℹ️  $node is local; skipping code sync"
            continue
        fi
        echo "   Syncing to $node..."
        rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' \
            "$PROJECT_DIR/" "$node:$REMOTE_PROJECT_DIR/" || {
            echo "❌ Error: Failed to sync code to $node"
            exit 1
        }
        echo "   ✅ Code synced to $node"
    done
else
    for node in "${ALL_NODES[@]}"; do
        if node_is_local "$node"; then
            echo "   [DRY RUN] $node is local; skipping code sync"
            continue
        fi
        echo "   [DRY RUN] rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' $PROJECT_DIR/ $node:$REMOTE_PROJECT_DIR/"
    done
fi

# Check SSH connectivity and remote requirements
echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${ALL_NODES[@]}"; do
        if ! node_check "$node"; then
            echo "❌ Error: Cannot connect to $node via SSH. Please check SSH setup."
            exit 1
        fi
        
        # Check if tmux is installed on remote node
        if ! node_exec "$node" "which tmux"; then
            echo "❌ Error: tmux is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi
        
        # Check if uv is installed on remote node
        if ! node_exec "$node" "uv --version"; then
            echo "❌ Error: uv is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi
        
        # Check if Promtail is installed on remote node (cross-platform)
        if node_exec "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && (promtail --version || promtail.exe --version || which promtail || where promtail.exe || test -f /c/promtail/promtail.exe || test -f /mnt/c/promtail/promtail.exe || test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" || test -f \"C:\\\\promtail\\\\promtail.exe\")" &>/dev/null; then
            # Kill any existing Promtail processes (cleanup old/broken instances)
            echo "🧹 $node: Cleaning up any existing Promtail processes and old logs..."
            node_exec "$node" "((pkill -f '[p]romtail' 2>/dev/null || (command -v sudo >/dev/null 2>&1 && sudo -n pkill -f '[p]romtail' 2>/dev/null) || true); (taskkill /F /IM promtail.exe >/dev/null 2>&1 || true))" &>/dev/null || true
            
            # Delete old log files and position files for fresh start
            node_exec "$node" "(rm -f $REMOTE_PROJECT_DIR/logging/cluster-logs/*.log /tmp/promtail-positions.yaml /tmp/positions.yaml 2>/dev/null || (command -v sudo >/dev/null 2>&1 && sudo -n rm -f $REMOTE_PROJECT_DIR/logging/cluster-logs/*.log /tmp/promtail-positions.yaml /tmp/positions.yaml 2>/dev/null) || true)" &>/dev/null || true
            
            # Ensure log directory exists
            node_exec "$node" "mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs"
            sleep 1
            
            # Determine config file based on node type
            if [[ "$node" == "$SERVER" ]]; then
                config_file="logging/promtail-server-remote.yaml"
            else
                config_file="logging/promtail-worker-remote.yaml"
            fi
            
            # Start Promtail in background (auto-detect path)
            echo "🚀 $node: Starting Promtail..."
            node_exec "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && PROMTAIL_CMD=\$(command -v promtail || command -v promtail.exe || (test -f /c/promtail/promtail.exe && echo /c/promtail/promtail.exe) || (test -f /mnt/c/promtail/promtail.exe && echo /mnt/c/promtail/promtail.exe) || (test -f \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\" && echo \"/c/Program Files/GrafanaLabs/Promtail/promtail.exe\") || (test -f \"C:\\\\promtail\\\\promtail.exe\" && echo \"C:\\\\promtail\\\\promtail.exe\") || echo promtail.exe) && nohup \$PROMTAIL_CMD -config.file=\$HOME/Desktop/smolcluster/$config_file > /tmp/promtail.log 2>&1 </dev/null &" &
            sleep 2
            
            # Check if Promtail is running
            if node_exec "$node" "pgrep -f promtail || tasklist /FI 'IMAGENAME eq promtail.exe' 2>nul | findstr promtail"; then
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
        if ! node_exec "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "⚠️  Venv not found on $node. Creating with Python 3.10..."
            node_exec "$node" "cd $REMOTE_PROJECT_DIR && uv venv --python 3.10 .venv && source .venv/bin/activate && uv pip install -e ."
        else
            echo "✅ Venv exists on $node. Running uv sync..."
            node_exec "$node" "cd $REMOTE_PROJECT_DIR && uv sync"
        fi
        
        # Special handling for Jetson devices - install CUDA-enabled PyTorch
        if [[ "$node" == *"jetson"* ]]; then
            echo "🤖 Detected Jetson device: $node"
            echo "   Installing Jetson-specific PyTorch with CUDA support..."
            node_exec "$node" "cd $REMOTE_PROJECT_DIR && bash scripts/installations/setup_jetson.sh"
            echo "   ✅ Jetson PyTorch installation complete"
        fi
        
        echo "✅ $node: SSH OK, tmux OK, uv OK, venv OK"
    done
else
    echo "✅ SSH and remote checks skipped (dry run)"
fi



echo "Server: $SERVER"
echo "Workers: ${WORKERS[*]}"
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
        echo "   [DRY RUN] Would launch $session_name on $node"
        return 0
    fi

    # SSH command with tmux and logging
    log_file="$REMOTE_PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"
    node_exec "$node" "cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\"" || {
        echo "❌ Failed to launch on $node"
        return 1
    }

    echo "✅ Launched $session_name on $node (logs: $log_file)"
    
    # Give tmux a moment to start
    sleep 1
    
    # Verify session exists
    if ! node_exec "$node" "tmux has-session -t $session_name "; then
        echo "⚠️  Warning: Session $session_name on $node may have exited. Check logs: $(node_attach_hint "$node" "$session_name")"
    fi
}


# Kill any existing sessions
echo ""
echo "🧹 Cleaning up existing sessions..."
if [[ "$DRY_RUN" != "true" ]]; then
    node_exec "$SERVER" "tmux kill-session -t server  || true"
    for worker_node in "${WORKERS[@]}"; do
        # Kill any session that starts with "worker"
        node_exec "$worker_node" "tmux list-sessions -F '#{session_name}'  | grep -E '^worker' | xargs -I {} tmux kill-session -t {}  || true"
    done
    echo "✅ Cleanup complete"
else
    echo "✅ Cleanup skipped (dry run)"
fi

# Launch server on $SERVER
echo ""
echo "🖥️  Launching server on $SERVER..."
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py server $SERVER --algorithm edp --resume-checkpoint '$RESUME_CHECKPOINT'"
else
    SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py server $SERVER --algorithm edp"
fi
launch_on_node "$SERVER" "$SERVER_CMD" "server"

# Wait a moment for server to start
echo "⏳ Waiting 2 seconds for server to initialize..."
sleep 2

# Launch workers
echo ""
echo "👷 Launching workers..."
for ((i=1; i<=NUM_WORKERS; i++)); do
    node="${WORKERS[$((i-1))]}"  # Get worker hostname by index
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py worker $i $node --algorithm edp --resume-checkpoint '$RESUME_CHECKPOINT'"
    else
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py worker $i $node --algorithm edp"
    fi
    launch_on_node "$node" "$WORKER_CMD" "worker$i"
    echo "   $node: worker$i"
done

echo ""
echo "🎉 Launch complete!"
echo ""
echo "📊 Check status:"
echo "   $(node_list_hint "$SERVER")"
echo "   $(node_attach_hint "$SERVER" "server")"
echo ""
echo "📈 Monitor training at: https://wandb.ai"