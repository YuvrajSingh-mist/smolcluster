#!/bin/bash

# GRPO training launcher with preflight checks and vLLM worker health checks.
# Usage:
#   ./scripts/training/launch_grpo_train.sh
#   ./scripts/training/launch_grpo_train.sh --dry-run
#   ./scripts/training/launch_grpo_train.sh --cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find project root by searching upward for pyproject.toml or .git
find_project_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/pyproject.toml" ]] || [[ -d "$dir/.git" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_DIR=$(find_project_root "$SCRIPT_DIR")
if [[ -z "$PROJECT_DIR" ]]; then
    echo "Error: could not find project root (looking for pyproject.toml or .git)"
    exit 1
fi

GRPO_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/reasoning/grpo/config.yaml"
SESSION_NAME="grpo_train"

DRY_RUN=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help|-h)
            echo "GRPO training launcher"
            echo ""
            echo "Options:"
            echo "  --dry-run    Print commands without executing"
            echo "  --cleanup    Kill the local tmux session used for GRPO training"
            echo "  --help, -h   Show help"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

if [[ "$CLEANUP" == "true" ]]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo "Killed tmux session: $SESSION_NAME"
    else
        echo "No tmux session found: $SESSION_NAME"
    fi
    exit 0
fi

require_cmd() {
    local cmd="$1"
    echo -n "Checking for required command: $cmd ... "
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found: $cmd"
        exit 1
    fi
    echo "OK"
}

echo "Verifying required commands:"
require_cmd yq
require_cmd curl
require_cmd tmux
require_cmd ssh
require_cmd uv
echo "All required commands found."

if [[ ! -f "$GRPO_CONFIG" ]]; then
    echo "Error: GRPO config not found: $GRPO_CONFIG"
    exit 1
fi

VLLM_ENABLED=$(yq '.vllm' "$GRPO_CONFIG")
if [[ "$VLLM_ENABLED" != "true" ]]; then
    echo "Error: vllm must be enabled in GRPO config to use this launcher"
    exit 1
fi

NUM_WORKERS=$(yq '.vllm_cluster.num_workers' "$GRPO_CONFIG")
TOTAL_NUM_NODES=$(yq '.vllm_cluster.total_num_nodes' "$GRPO_CONFIG")
SERVER_HOST=$(yq '.vllm_cluster.server' "$GRPO_CONFIG")
VLLM_PORT=$(yq '.vllm_cluster.port' "$GRPO_CONFIG")
COMPLETION_PATH=$(yq '.vllm_cluster.completion_path' "$GRPO_CONFIG")

WORKER_ENTRIES=()
while IFS= read -r entry; do
    [[ -n "$entry" ]] && WORKER_ENTRIES+=("$entry")
done < <(yq '.vllm_cluster.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$GRPO_CONFIG")

WORKER_HOSTS=()
WORKER_RANKS=()
for entry in "${WORKER_ENTRIES[@]}"; do
    WORKER_HOSTS+=("${entry%%:*}")
    WORKER_RANKS+=("${entry##*:}")
done

if [[ ${#WORKER_HOSTS[@]} -ne "$NUM_WORKERS" ]]; then
    echo "Error: vllm_cluster.num_workers ($NUM_WORKERS) does not match worker entries (${#WORKER_HOSTS[@]})"
    exit 1
fi

EXPECTED_TOTAL=$((NUM_WORKERS + 1))
if [[ "$TOTAL_NUM_NODES" -ne "$EXPECTED_TOTAL" ]]; then
    echo "Error: vllm_cluster.total_num_nodes ($TOTAL_NUM_NODES) must equal num_workers + 1 ($EXPECTED_TOTAL)"
    exit 1
fi

echo "Preflight checks"
echo "  Config: $GRPO_CONFIG"
echo "  Server: $SERVER_HOST"
echo "  Workers: ${WORKER_HOSTS[*]}"
echo "  Port/path: $VLLM_PORT$COMPLETION_PATH"

if [[ "$DRY_RUN" == "false" ]]; then
    for worker in "${WORKER_HOSTS[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$worker" "echo ok" >/dev/null 2>&1; then
            echo "Error: cannot SSH to worker node: $worker"
            exit 1
        fi
    done

    for i in "${!WORKER_HOSTS[@]}"; do
        worker="${WORKER_HOSTS[$i]}"
        rank="${WORKER_RANKS[$i]}"
        worker_ip=$(yq ".vllm_cluster.host_ip.${worker}" "$GRPO_CONFIG")
        if [[ -z "$worker_ip" || "$worker_ip" == "null" ]]; then
            echo "Error: missing host_ip mapping for worker: $worker"
            exit 1
        fi

        completion_url="http://${worker_ip}:${VLLM_PORT}${COMPLETION_PATH}"
        models_url="http://${worker_ip}:${VLLM_PORT}/v1/models"

        echo "Checking worker rank $rank ($worker) at $completion_url"

        if ! curl -sS --max-time 8 "$models_url" >/dev/null; then
            echo "Error: vLLM /v1/models not reachable on $worker ($models_url)"
            exit 1
        fi

        payload='{"prompt":"ping","max_tokens":1,"temperature":0.0}'
        if ! curl -sS --max-time 12 -H "Content-Type: application/json" -d "$payload" "$completion_url" | yq '.choices[0].text' >/dev/null 2>&1; then
            echo "Error: vLLM completion check failed on $worker ($completion_url)"
            exit 1
        fi

        echo "  OK: vLLM reachable on $worker"
    done
else
    echo "Dry run enabled: skipping SSH and vLLM endpoint checks"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Existing tmux session found: $SESSION_NAME"
    if [[ "$DRY_RUN" == "false" ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "Killed existing session: $SESSION_NAME"
    else
        echo "Dry run: would kill existing session"
    fi
fi

TRAIN_CMD="cd $PROJECT_DIR && uv run python -m smolcluster.applications.reasoning.grpo.train"

echo "Launching GRPO training on server node ($SERVER_HOST)"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run command: tmux new -d -s $SESSION_NAME \"bash -lc '$TRAIN_CMD'\""
else
    tmux new -d -s "$SESSION_NAME" "bash -lc '$TRAIN_CMD'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
fi
