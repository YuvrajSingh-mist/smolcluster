#!/bin/bash

# set -e

# Load environment variables from .env
if [[ -f "../../.env" ]]; then
    export $(grep -v '^#' ../../.env | xargs)
elif [[ -f ".env" ]]; then
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
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config_fsdp.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"  # Adjust if your remote path is different

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/node_helpers.sh"
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/logging_helpers.sh"
init_node_helpers "$CONFIG_FILE" "$PROJECT_DIR" "$REMOTE_PROJECT_DIR"

# Log in to HuggingFace Hub so gated models (e.g. Llama-2) can be downloaded.
ensure_hf_login_local

# Read configuration from YAML
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")

# Read regular workers (hostname and rank) - bash 3.2 compatible
REGULAR_WORKERS=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && REGULAR_WORKERS+=("$worker")
done < <(yq '.allToAllTopology.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE")

# Read tablet workers (hostname and rank) - bash 3.2 compatible
TABLET_WORKERS=()
while IFS= read -r tablet; do
    [[ -n "$tablet" ]] && TABLET_WORKERS+=("$tablet")
done < <(yq '.allToAllTopology.workers.tablets[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null || true)

# Extract just hostnames for SSH operations
WORKERS=()
for worker in "${REGULAR_WORKERS[@]}"; do
    [[ -n "$worker" ]] && WORKERS+=("${worker%%:*}")
done
TABLETS=()
for tablet in "${TABLET_WORKERS[@]}"; do
    [[ -n "$tablet" ]] && TABLETS+=("${tablet%%:*}")
done
ALL_NODES=("${WORKERS[@]}" "${TABLETS[@]}")

# Validate configuration
ACTUAL_WORKER_COUNT=$((${#WORKERS[@]} + ${#TABLETS[@]}))
if [[ $ACTUAL_WORKER_COUNT -ne $NUM_WORKERS ]]; then
    echo "❌ Error: num_workers ($NUM_WORKERS) does not match total workers (${#WORKERS[@]} regular + ${#TABLETS[@]} tablets = $ACTUAL_WORKER_COUNT)"
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

# Read FSDP stage from config
FSDP_STAGE=$(yq '.fsdp_stage // 1' "$CONFIG_FILE")
FSDP_STAGE_NAME="ZeRO Stage 0 (Optimizer Partitioning)"
if [[ $FSDP_STAGE -eq 1 ]]; then
    FSDP_STAGE_NAME="ZeRO Stage 1 (Optimizer + Gradient Partitioning)"
elif [[ $FSDP_STAGE -eq 2 ]]; then
    FSDP_STAGE_NAME="ZeRO Stage 2 (Optimizer + Gradient + Parameter Partitioning)"
fi

echo "🚀 SmolCluster Launch Script - FSDP Stage $FSDP_STAGE ($FSDP_STAGE_NAME)"
echo "📁 Project dir: $PROJECT_DIR"
echo "⚙️  Config file: $CONFIG_FILE"

echo "📤 This API key will be used on all remote nodes"

# Create array of nodes that need SSH (all workers, not tablets)
SSH_NODES=("${WORKERS[@]}")

# Sync code to all remote nodes first
echo "📦 Syncing code to remote nodes..."
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
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
    echo "✅ Code sync complete"
else
    echo "✅ Code sync skipped (dry run)"
fi

# Check SSH connectivity and remote requirements
echo "🔗 Checking SSH connectivity and remote requirements..."
if [[ ${#TABLETS[@]} -gt 0 ]]; then
    echo "ℹ️  Skipping SSH checks for tablets: ${TABLETS[*]} (run locally on device)"
fi
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
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
        
        # Ensure log directory exists on remote node
        node_exec "$node" "mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs"
        
        # Check that venv exists and sync dependencies
        echo "📦 Checking venv on $node..."
        if ! node_exec "$node" "test -f $REMOTE_PROJECT_DIR/.venv/bin/python"; then
            echo "⚠️  Venv not found on $node. Creating with Python 3.10..."
            node_exec "$node" "cd $REMOTE_PROJECT_DIR && uv venv --python 3.10 .venv && source .venv/bin/activate && uv pip install -e ."
        else
            echo "✅ Venv exists on $node. Running uv sync..."
            node_exec "$node" "cd $REMOTE_PROJECT_DIR && uv sync"
        fi
        
        # Check Python environment and required libraries
        echo "🔍 $node: Checking Python environment and required libraries..."
        
        # Run comprehensive pre-check script using heredoc
        PRECHECK_RESULT=$(node_bash "$node" <<'ENDSSH'
export PATH=/opt/homebrew/bin:/usr/local/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
cd ~/Desktop/smolcluster
source .venv/bin/activate
python <<'ENDPYTHON'
import sys
import importlib.util

def check_package(package_name, min_version=None):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, None
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, version
        return True, version
    except ImportError:
        return False, None

# Check required packages
required_packages = {
    "torch": "2.0.0",
    "numpy": "1.24.0",
    "transformers": "4.57.5",
    "datasets": "4.4.2",
    "tqdm": None,
    "wandb": "0.17.0",
    "yaml": None
}

missing_packages = []
version_mismatches = []

for package, min_ver in required_packages.items():
    found, version = check_package(package, min_ver)
    if not found:
        if version:
            version_mismatches.append(f"{package} (found {version}, need >={min_ver})")
        else:
            missing_packages.append(package)

# Check CUDA availability in PyTorch
import torch
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else None

# Print results
if missing_packages or version_mismatches:
    print("MISSING_DEPS:", ",".join(missing_packages + version_mismatches))
    sys.exit(1)
else:
    print(f"ALL_OK:CUDA={cuda_available}:CUDA_VERSION={cuda_version}:TORCH={torch.__version__}:PYTHON={sys.version.split()[0]}")
    sys.exit(0)
ENDPYTHON
ENDSSH
)
        
        PRECHECK_EXIT=$?
        
        if [[ $PRECHECK_EXIT -eq 0 ]]; then
            # Parse the pre-check result
            if [[ "$PRECHECK_RESULT" == ALL_OK:* ]]; then
                echo "✅ $node: All dependencies satisfied"
                echo "   $PRECHECK_RESULT" | sed 's/ALL_OK://; s/:/  /g'
                
                # Check if it's a Jetson device with failed CUDA
                if [[ "$node" == *"jetson"* ]] && [[ "$PRECHECK_RESULT" != *"CUDA=True"* ]]; then
                    echo "⚠️  $node: Jetson device detected but CUDA not available in PyTorch"
                    echo "   Running Jetson-specific PyTorch installation..."
                    node_exec "$node" "cd $REMOTE_PROJECT_DIR && bash scripts/installations/setup_jetson.sh"
                    echo "   ✅ Jetson PyTorch installation complete"
                fi
            fi
        else
            echo "⚠️  $node: Pre-check failed - missing dependencies or version mismatches"
            echo "   $PRECHECK_RESULT"
            
            # For Jetson devices, run the setup script
            if [[ "$node" == *"jetson"* ]]; then
                echo "🤖 $node: Running Jetson-specific installation..."
                node_exec "$node" "cd $REMOTE_PROJECT_DIR && bash scripts/installations/setup_jetson.sh"
                echo "   ✅ Jetson installation attempt complete"
            else
                echo "❌ $node: Please install missing dependencies manually"
                echo "   Run: $(node_command_hint "$node" "cd $REMOTE_PROJECT_DIR && source .venv/bin/activate && uv pip install -e .")"
                exit 1
            fi
        fi
        
        echo "✅ $node: SSH OK, tmux OK, uv OK, venv OK, dependencies OK"
    done
else
    echo "✅ SSH and remote checks skipped (dry run)"
fi



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
        echo "   [DRY RUN] Would launch on $node"
        return 0
    fi

    log_file="$REMOTE_PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"
    node_exec "$node" "cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\"" || {
        echo "❌ Failed to launch on $node"
        return 1
    }
    echo "✅ Launched $session_name on $node (logs: $log_file)"

    # Stream remote log to controller so the dashboard log tab can show it.
    if ! node_is_local "$node"; then
        local _local_log="$PROJECT_DIR/logging/cluster-logs/${session_name}__${node}.log"
        rm -f "$_local_log"
        ( sleep 2; ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
            -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
            "$node" "tail -F $log_file 2>/dev/null" >> "$_local_log" 2>/dev/null ) &
        disown $! 2>/dev/null || true
    fi

    # Give tmux a moment to start
    sleep 1
    
    # Verify session exists
    if ! node_exec "$node" "tmux has-session -t $session_name"; then
        echo "⚠️  Warning: Session $session_name on $node may have exited. Check logs: $(node_attach_hint "$node" "$session_name")"
    fi
}


# Kill any existing sessions
echo ""
echo "🧹 Cleaning up existing sessions..."
if [[ "$DRY_RUN" != "true" ]]; then
    for worker_node in "${WORKERS[@]}"; do
        node_exec "$worker_node" "tmux list-sessions -F '#{session_name}' | grep -E '^fsdp_worker' | xargs -I {} tmux kill-session -t {} || true"
    done
    echo "✅ Cleanup complete"
else
    echo "✅ Cleanup skipped (dry run)"
fi


# Launch workers
echo ""
echo "👷 Launching workers..."

# Find worker with rank 0 and launch it first
WORKER_0_ENTRY=""
WORKER_0_HOSTNAME=""
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    rank="${worker_entry##*:}"
    if [[ "$rank" == "0" ]]; then
        WORKER_0_ENTRY="$worker_entry"
        WORKER_0_HOSTNAME="${worker_entry%%:*}"
        break
    fi
done

if [[ -z "$WORKER_0_HOSTNAME" ]]; then
    echo "❌ Error: No worker with rank 0 found in config"
    exit 1
fi

echo ""
echo "🖥️  Launching worker rank 0 on $WORKER_0_HOSTNAME..."
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    WORKER_0_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py worker 0 $WORKER_0_HOSTNAME --algorithm fsdp --resume-checkpoint '$RESUME_CHECKPOINT'"
else
    WORKER_0_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py worker 0 $WORKER_0_HOSTNAME --algorithm fsdp"
fi
launch_on_node "$WORKER_0_HOSTNAME" "$WORKER_0_CMD" "fsdp_worker0"
echo "   ✅ Rank 0: $WORKER_0_HOSTNAME (fsdp_worker0)"


if [[ ${#TABLET_WORKERS[@]} -gt 0 ]]; then
    echo "ℹ️  Tabletsould run manually: "
    for worker_entry in "${TABLET_WORKERS[@]}"; do
        hostname="${worker_entry%%:*}"
        rank="${worker_entry##*:}"
        echo "      $hostname: python worker_tablets.py $rank $hostname"
    done
fi

# Launch remaining workers (skip rank 0, already launched)
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    rank="${worker_entry##*:}"
    
    # Skip worker rank 0 (already launched)
    if [[ "$rank" == "0" ]]; then
        continue
    fi
    
    # Launch regular worker via SSH
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py worker $rank $hostname --algorithm fsdp --resume-checkpoint '$RESUME_CHECKPOINT'"
    else
        WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && cd src/smolcluster && ../../.venv/bin/python train.py worker $rank $hostname --algorithm fsdp"
    fi
    launch_on_node "$hostname" "$WORKER_CMD" "fsdp_worker$rank"
    echo "   ✅ Rank $rank: $hostname (fsdp_worker$rank)"
done

# Launch tablet workers (manual reminder only - they're already in the list above)
if [[ ${#TABLET_WORKERS[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  Remember to manually start tablet workers asown above"
fi

echo ""
echo "🎉 Launch complete!"
echo ""
echo "📊 Check status:"
for worker_node in "${WORKERS[@]}"; do
    echo "   $(node_list_hint "$worker_node")"
done
echo "   $(node_attach_hint "${WORKERS[0]}" "fsdp_worker0")"
echo ""
echo "📈 Monitor training at: https://wandb.ai"
