#!/bin/bash

# SmolCluster Launch Script
# Launches distributed training across Mac mini nodes via SSH

set -e  # Exit on any error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"  # Adjust if your remote path is different

# Check for dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🏃 Dry run mode - will show commands without executing"
fi

echo "🚀 SmolCluster Launch Script"
echo "📁 Project dir: $PROJECT_DIR"
echo "⚙️  Config file: $CONFIG_FILE"


# Check SSH connectivity and remote requirements
echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${ALL_NODES[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'SSH OK'"; then
            echo "❌ Error: Cannot connect to $node via SSH. Please check SSH setup."
            exit 1
        fi
        
        # Check if tmux is installed on remote node
        if ! ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && which tmux"; then
            echo "❌ Error: tmux is not installed on $node. Install with: ssh $node 'brew install tmux'"
            exit 1
        fi
        
        # Check if uv is installed on remote node
        if ! ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && uv --version"; then
            echo "❌ Error: uv is not installed on $node. Install with: ssh $node 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
            exit 1
        fi
        
        # Check that venv exists (don't run uv sync as it resets Python version)
        echo "📦 Checking venv on $node..."
        if ! ssh "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "⚠️  Venv not found on $node. Creating with Python 3.9..."
            ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv venv --python 3.9.6 .venv && source .venv/bin/activate && uv pip install -e ."
        fi
        
        echo "✅ $node: SSH OK, tmux OK, uv OK, venv OK"
    done
else
    echo "✅ SSH and remote checks skipped (dry run)"
fi

# Read configuration from YAML
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")
SERVER=$(yq '.server' "$CONFIG_FILE")
WORKERS=($(yq '.workers[]' "$CONFIG_FILE"))
ALL_NODES=("$SERVER" "${WORKERS[@]}")

echo "Server: $SERVER"
echo "Workers: ${WORKERS[*]}"
echo "All nodes: ${ALL_NODES[*]}"

# Function to launch on a node
launch_on_node() {
    local node=$1
    local command=$2
    local session_name=$3

    echo "🔗 Launching on $node: $command"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_file="\$HOME/${session_name}.log"
        echo "   [DRY RUN] Would execute: ssh $node \"export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && /opt/homebrew/bin/tmux new -d -s $session_name \\\"bash -c '$command 2>&1 | tee $log_file; exec bash'\\\"\""
        return 0
    fi

    # SSH command with tmux and logging
    log_file="\$HOME/${session_name}.log"
    ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && /opt/homebrew/bin/tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\"" || {
        echo "❌ Failed to launch on $node"
        return 1
    }

    echo "✅ Launched $session_name on $node (logs: $log_file)"
    
    # Give tmux a moment to start
    sleep 1
    
    # Verify session exists
    if ! ssh "$node" "/opt/homebrew/bin/tmux has-session -t $session_name 2>/dev/null"; then
        echo "⚠️  Warning: Session $session_name on $node may have exited. Check logs: ssh $node 'tail -20 $log_file'"
    fi
}


# Kill any existing sessions
echo ""
echo "🧹 Cleaning up existing sessions..."
if [[ "$DRY_RUN" != "true" ]]; then
    ssh "$SERVER" "/opt/homebrew/bin/tmux kill-session -t server 2>/dev/null || true"
    for worker_node in "${WORKERS[@]}"; do
        # Kill any session that starts with "worker"
        ssh "$worker_node" "/opt/homebrew/bin/tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^worker' | xargs -I {} /opt/homebrew/bin/tmux kill-session -t {} 2>/dev/null || true"
    done
    echo "✅ Cleanup complete"
else
    echo "✅ Cleanup skipped (dry run)"
fi

# Launch server on $SERVER
echo ""
echo "🖥️  Launching server on $SERVER..."
SERVER_CMD="cd src/smolcluster/SimpleAllReduce && ../../../.venv/bin/python server.py"
launch_on_node "$SERVER" "$SERVER_CMD" "server"

# Wait a moment for server to start
echo "⏳ Waiting 5 seconds for server to initialize..."
sleep 5

# Launch workers
echo ""
echo "👷 Launching workers..."
for ((i=1; i<=NUM_WORKERS; i++)); do
    node="${WORKERS[$((i-1))]}"  # Get worker hostname by index
    WORKER_CMD="cd src/smolcluster/SimpleAllReduce && ../../../.venv/bin/python worker.py $i"
    launch_on_node "$node" "$WORKER_CMD" "worker$i"
    echo "   $node: worker$i"
done

echo ""
echo "🎉 Launch complete!"
echo ""
echo "📊 Check status:"
echo "   ssh $SERVER 'tmux ls'"
echo "   ssh $SERVER 'tmux attach -t server'"
echo ""
echo "📈 Monitor training at: https://wandb.ai"