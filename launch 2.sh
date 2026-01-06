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
    for node in mini1 mini2 mini3; do
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
        
        # Install project dependencies on remote node
        echo "📦 Installing project dependencies on $node..."
        if ! ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv sync"; then
            echo "❌ Error: Failed to install project on $node"
            exit 1
        fi
        
        echo "✅ $node: SSH OK, tmux OK, uv OK, dependencies OK"
    done
else
    echo "✅ SSH and remote checks skipped (dry run)"
fi

# Read number of workers from config
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")
echo "Workers configured: $NUM_WORKERS"

# Function to launch on a node
launch_on_node() {
    local node=$1
    local command=$2
    local session_name=$3

    echo "🔗 Launching on $node: $command"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   [DRY RUN] Would execute: ssh $node \"export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && export WANDB_API_KEY='$WANDB_API_KEY' && cd $REMOTE_PROJECT_DIR && uv run wandb login --relogin '$WANDB_API_KEY' && tmux new -d -s $session_name '$command'\""
        return 0
    fi

    # SSH command with W&B login and tmux
    ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && export WANDB_API_KEY='$WANDB_API_KEY' && cd $REMOTE_PROJECT_DIR && uv run wandb login --relogin '$WANDB_API_KEY' && tmux new -d -s $session_name '$command'"|| {
        echo "❌ Failed to launch on $node"
        return 1
    }

    echo "✅ Launched $session_name on $node"
}


# Launch server on mini1
echo ""
echo "🖥️  Launching server on mini1..."
SERVER_CMD="uv run python src/smolcluster/NoRingReduce/server.py"
launch_on_node "mini1" "$SERVER_CMD" "server"

# Wait a moment for server to start
echo "⏳ Waiting 3 seconds for server to initialize..."
sleep 3

# Launch workers
echo ""
echo "👷 Launching workers..."
for ((i=1; i<=NUM_WORKERS; i++)); do
    node="mini$((i+1))"  # mini2, mini3, etc.
    WORKER_CMD="uv run python src/smolcluster/NoRingReduce/worker.py"
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