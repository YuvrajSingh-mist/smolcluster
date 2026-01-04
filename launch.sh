#!/bin/bash

# SmolCluster Launch Script
# Launches distributed training across Mac mini nodes via SSH

set -e  # Exit on any error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config.yaml"
REMOTE_PROJECT_DIR="~/smolcluster"  # Adjust if your remote path is different

# Check for dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🏃 Dry run mode - will show commands without executing"
fi

echo "🚀 SmolCluster Launch Script"
echo "📁 Project dir: $PROJECT_DIR"
echo "⚙️  Config file: $CONFIG_FILE"

# Pre-flight checks
echo ""
echo "🔍 Running pre-flight checks..."

# Check if yq is installed (needed locally)
if ! command -v yq &> /dev/null; then
    echo "❌ Error: yq is required to parse YAML. Install with: brew install yq"
    exit 1
fi
echo "✅ yq found"

# Check W&B API key
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "❌ Error: WANDB_API_KEY environment variable is not set."
    echo "   Set it with: export WANDB_API_KEY='your_api_key_here'"
    echo "   Get your API key from: https://wandb.ai/settings"
    exit 1
fi
echo "✅ W&B API key found"

# Test W&B login locally
echo "🔑 Testing W&B login..."
if [[ "$DRY_RUN" != "true" ]]; then
    if ! wandb login --relogin "$WANDB_API_KEY" 2>/dev/null; then
        echo "❌ Error: Failed to login to W&B. Please check your API key."
        exit 1
    fi
    echo "✅ W&B login successful"
else
    echo "✅ W&B login skipped (dry run)"
fi

# Check SSH connectivity and remote requirements
echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in mini1 mini2 mini3; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'SSH OK'" >/dev/null 2>&1; then
            echo "❌ Error: Cannot connect to $node via SSH. Please check SSH setup."
            exit 1
        fi
        
        # Check if tmux is installed on remote node
        if ! ssh "$node" "command -v tmux" >/dev/null 2>&1; then
            echo "❌ Error: tmux is not installed on $node. Install with: ssh $node 'brew install tmux'"
            exit 1
        fi
        
        # Check if wandb is installed on remote node
        if ! ssh "$node" "command -v wandb" >/dev/null 2>&1; then
            echo "❌ Error: wandb is not installed on $node. Install with: ssh $node 'pip install wandb'"
            exit 1
        fi
        
        echo "✅ $node: SSH OK, tmux OK, wandb OK"
    done
else
    echo "✅ SSH and remote checks skipped (dry run)"
fi

# Read number of workers from config
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")
echo "👥 Workers configured: $NUM_WORKERS"

# Function to launch on a node
launch_on_node() {
    local node=$1
    local command=$2
    local session_name=$3

    echo "🔗 Launching on $node: $command"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   [DRY RUN] Would execute: ssh $node \"export WANDB_API_KEY='$WANDB_API_KEY' && wandb login --relogin '$WANDB_API_KEY' && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name '$command'\""
        return 0
    fi

    # SSH command with W&B login and tmux
    ssh "$node" "export WANDB_API_KEY='$WANDB_API_KEY' && wandb login --relogin '$WANDB_API_KEY' && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name '$command'" 2>/dev/null || {
        echo "❌ Failed to launch on $node"
        return 1
    }

    echo "✅ Launched $session_name on $node"
}

# Launch server on mini1
echo ""
echo "🖥️  Launching server on mini1..."
SERVER_CMD="cd src/smolcluster && uv run python NoRingReduce/server.py"
launch_on_node "mini1" "$SERVER_CMD" "server"

# Wait a moment for server to start
echo "⏳ Waiting 3 seconds for server to initialize..."
sleep 3

# Launch workers
echo ""
echo "👷 Launching workers..."
for ((i=1; i<=NUM_WORKERS; i++)); do
    node="mini$((i+1))"  # mini2, mini3, etc.
    WORKER_CMD="cd src/smolcluster && uv run python NoRingReduce/worker.py"
    launch_on_node "$node" "$WORKER_CMD" "worker$i"
done

echo ""
echo "🎉 Launch complete!"
echo ""
echo "📊 Check status:"
echo "   ssh mini1 'tmux ls'"
echo "   ssh mini1 'tmux attach -t server'"
echo "   ssh mini2 'tmux attach -t worker1'"
echo ""
echo "🛑 To stop all:"
echo "   ssh mini1 'tmux kill-session -t server'"
echo "   ssh mini2 'tmux kill-session -t worker1'"
echo "   ssh mini3 'tmux kill-session -t worker2'"
echo ""
echo "📈 Monitor training at: https://wandb.ai"