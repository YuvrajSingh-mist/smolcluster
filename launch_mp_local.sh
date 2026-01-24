#!/bin/bash

# Simple local launcher for Model Parallelism on localhost

# Load environment variables from .env
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set WANDB_API_KEY for wandb compatibility
export WANDB_API_KEY="$WANDB_API_TOKEN"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config_mp.yaml"

echo "🚀 SmolCluster Local Launch - Model Parallelism"
echo "📁 Project dir: $PROJECT_DIR"
echo "⚙️  Config file: $CONFIG_FILE"

# Kill any existing processes on port 65432
echo "🧹 Cleaning up any existing processes on port 65432..."
lsof -ti:65432 | xargs kill -9 2>/dev/null || true
sleep 1

# Kill any existing tmux sessions
echo "🧹 Killing any existing smolcluster tmux sessions..."
tmux kill-session -t mp-server 2>/dev/null || true
tmux kill-session -t mp-worker-1 2>/dev/null || true
tmux kill-session -t mp-worker-2 2>/dev/null || true
sleep 1

echo ""
echo "🖥️  Starting Server (rank 0)..."
tmux new-session -d -s mp-server "cd '$PROJECT_DIR/src/smolcluster' && ../../.venv/bin/python train.py server localhost --algorithm mp 2>&1 | tee /tmp/mp-server.log; exec bash"

sleep 3

echo "👷 Starting Worker 1 (rank 1)..."
tmux new-session -d -s mp-worker-1 "cd '$PROJECT_DIR/src/smolcluster' && ../../.venv/bin/python train.py worker 1 localhost --algorithm mp 2>&1 | tee /tmp/mp-worker-1.log; exec bash"

sleep 2

echo "👷 Starting Worker 2 (rank 2)..."
tmux new-session -d -s mp-worker-2 "cd '$PROJECT_DIR/src/smolcluster' && ../../.venv/bin/python train.py worker 2 localhost --algorithm mp 2>&1 | tee /tmp/mp-worker-2.log; exec bash"

echo ""
echo "✅ All processes started!"
echo ""
echo "📊 Monitor with:"
echo "  Server:   tmux attach -t mp-server"
echo "  Worker 1: tmux attach -t mp-worker-1"
echo "  Worker 2: tmux attach -t mp-worker-2"
echo ""
echo "📝 Logs:"
echo "  tail -f /tmp/mp-server.log"
echo "  tail -f /tmp/mp-worker-1.log"
echo "  tail -f /tmp/mp-worker-2.log"
echo ""
echo "🛑 To stop all:"
echo "  tmux kill-session -t mp-server && tmux kill-session -t mp-worker-1 && tmux kill-session -t mp-worker-2"
