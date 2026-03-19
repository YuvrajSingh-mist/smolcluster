#!/bin/bash

# Launch ClassicDP inference workers (rank 0 is leader), then chat API/frontend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config_classicdp.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

export WANDB_API_KEY="$WANDB_API_TOKEN"

WORKERS=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && WORKERS+=("$worker")
done < <(yq '.allToAllTopology.workers.regular[] | .hostname + ":" + (.rank | tostring) + ":" + .ip + ":" + (.port | tostring)' "$CONFIG_FILE")

if [[ ${#WORKERS[@]} -eq 0 ]]; then
    echo "❌ No workers found in allToAllTopology.workers.regular"
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

RANK0_HOST=$(echo "$RANK0_ENTRY" | cut -d: -f1)
RANK0_IP=$(echo "$RANK0_ENTRY" | cut -d: -f3)
RANK0_PORT=$(echo "$RANK0_ENTRY" | cut -d: -f4)

SSH_NODES=()
for entry in "${WORKERS[@]}"; do
    hostname=$(echo "$entry" | cut -d: -f1)
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
echo "📁 Project: $PROJECT_DIR"
echo "⚙️  Config: $CONFIG_FILE"
echo "🧠 Leader (rank 0): $RANK0_HOST @ $RANK0_IP:$RANK0_PORT"

echo "📦 Syncing code to remote ClassicDP nodes"
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        echo "   Syncing to $node..."
        rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' \
            "$PROJECT_DIR/" "$node:$REMOTE_PROJECT_DIR/" || {
            echo "❌ Error: Failed to sync code to $node"
            exit 1
        }
        echo "   ✅ Code synced to $node"
    done
else
    for node in "${SSH_NODES[@]}"; do
        echo "   [DRY RUN] rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' $PROJECT_DIR/ $node:$REMOTE_PROJECT_DIR/"
    done
fi

echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'SSH OK'"; then
            echo "❌ Error: Cannot connect to $node via SSH. Please check SSH setup."
            exit 1
        fi

        if ! ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && which tmux >/dev/null 2>&1"; then
            echo "❌ Error: tmux is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi

        if ! ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && uv --version >/dev/null 2>&1"; then
            echo "❌ Error: uv is not installed on $node. Install deps on $node with: ssh $node 'bash $REMOTE_PROJECT_DIR/scripts/installations/installation.sh'"
            exit 1
        fi

        echo "📦 Checking venv on $node..."
        if ! ssh "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "⚠️  Venv not found on $node. Creating with Python 3.10..."
            ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv venv --python 3.10 .venv && uv pip install -e ."
        else
            echo "✅ Venv exists on $node. Running uv sync..."
            ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv sync"
        fi

        echo "🧪 Verifying smolcluster import on $node..."
        if ! ssh "$node" "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -c 'import smolcluster'"; then
            echo "⚠️  Import failed on $node after sync. Reinstalling editable package..."
            ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && uv pip install -e ."
            if ! ssh "$node" "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -c 'import smolcluster'"; then
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
    local log_file="\$HOME/${session_name}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        local safe_command="$command"
        safe_command=$(echo "$safe_command" | sed -E "s/WANDB_API_KEY='[^']*'/WANDB_API_KEY='***REDACTED***'/g; s/HF_TOKEN='[^']*'/HF_TOKEN='***REDACTED***'/g")
        echo "   [DRY RUN] ssh $node \"cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \\\"bash -c '$safe_command 2>&1 | tee $log_file; exec bash'\\\"\""
        return 0
    fi

    ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\""
}

echo "🧹 Cleaning up old sessions"
if [[ "$DRY_RUN" != "true" ]]; then
    for entry in "${WORKERS[@]}"; do
        hostname=$(echo "$entry" | cut -d: -f1)
        rank=$(echo "$entry" | cut -d: -f2)
        ssh "$hostname" "tmux kill-session -t classicdp_inf_worker$rank 2>/dev/null || true"
    done
fi

echo "👷 Starting ClassicDP inference workers"
for entry in "${WORKERS[@]}"; do
    hostname=$(echo "$entry" | cut -d: -f1)
    rank=$(echo "$entry" | cut -d: -f2)

    WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' PYTHONPATH='$REMOTE_PROJECT_DIR/src':\$PYTHONPATH && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/ClassicDP/inference/worker.py $rank $hostname"
    action_ssh "$hostname" "$WORKER_CMD" "classicdp_inf_worker$rank"
    echo "   ✅ Worker rank $rank on $hostname"
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
