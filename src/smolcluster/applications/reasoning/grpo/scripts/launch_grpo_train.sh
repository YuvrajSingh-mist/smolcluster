#!/bin/bash

# GRPO training launcher with preflight checks and vLLM worker health checks.
# Usage:
#   ./scripts/training/launch_grpo_train.sh [gsm8k|summarization]
#   ./scripts/training/launch_grpo_train.sh --dry-run [gsm8k|summarization]
#   ./scripts/training/launch_grpo_train.sh --cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$PROJECT_DIR/.env"
    set -u
fi

GRPO_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/reasoning/grpo/config.yaml"
CLUSTER_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
MODEL_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"
VLLM_TMUX_SESSION="vllm_worker"

DRY_RUN=false
CLEANUP=false
TRAIN_TARGET="summarization"  # default target

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true;  shift ;;
        --cleanup)  CLEANUP=true;  shift ;;
        --help|-h)
            echo "GRPO training launcher"
            echo ""
            echo "Options:"
            echo "  --dry-run    Print commands without executing"
            echo "  --cleanup    Stop GRPO training auxiliaries and all vLLM worker processes"
            echo "  --help, -h   Show help"
            echo ""
            echo "Training targets:"
            echo "  gsm8k"
            echo "  summarization"
            exit 0
            ;;
        gsm8k|summarization)
            TRAIN_TARGET="$1"
            shift
            ;;
        *)
            echo "Error: Unknown option or training target: $1"
            echo "Valid targets: gsm8k, summarization"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# PID file for background SSH log-tail processes (one per vLLM worker).
# Written during launch; read by --cleanup to stop the tails.
GRPO_TAIL_PIDS_FILE="/tmp/smolcluster_grpo_tail_pids_${TRAIN_TARGET}.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Prepended to every remote SSH command so tools installed outside default PATH are found.
REMOTE_PATH="export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"

require_cmd() {
    echo -n "  $1 ... "
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "MISSING"
        echo "Error: required command not found: $1"
        exit 1
    fi
    echo "ok"
}

# Kill the vLLM tmux session and free the port on a remote host.
kill_vllm_on_worker() {
    local host="$1" ip="$2" port="$3"
    echo "  [$host] killing tmux session '${VLLM_TMUX_SESSION}' and freeing port $port ..."
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" "
        ${REMOTE_PATH}
        tmux kill-session -t '${VLLM_TMUX_SESSION}' >/dev/null 2>&1 || true
        pkill -9 -f 'vllm serve' >/dev/null 2>&1 || true
        fuser -k ${port}/tcp >/dev/null 2>&1 || true
        lsof -ti :${port} 2>/dev/null | xargs kill -9 >/dev/null 2>&1 || true
        echo done
    " 2>/dev/null || echo "  [$host] ssh kill attempt finished (process may already have been gone)"
}

# Poll until the vLLM endpoint stops responding — confirmed dead.
wait_for_vllm_down() {
    local host="$1" ip="$2" port="$3"
    local url="http://${ip}:${port}/v1/models"
    local attempt=0 max=20
    echo "  [$host] waiting for vLLM to go down ..."
    while [[ $attempt -lt $max ]]; do
        if ! curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
            echo "  [$host] confirmed down ✓"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    echo "  [$host] WARNING: vLLM still responding after ${max}s — may not have shut down cleanly"
    return 1
}



# Kill vLLM on every worker, confirm dead, then verify memory is free.
# Uses globals: WORKER_HOSTS WORKER_IPS VLLM_PORT
reset_all_vllm_workers() {
    if [[ ${#WORKER_HOSTS[@]} -eq 0 ]]; then
        echo "  No workers configured, skipping vLLM reset."
        return 0
    fi
    echo "Killing vLLM on all workers ..."
    for i in "${!WORKER_HOSTS[@]}"; do
        kill_vllm_on_worker "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT"
    done
    echo "Waiting for all workers to go down ..."
    local all_ok=true
    for i in "${!WORKER_HOSTS[@]}"; do
        wait_for_vllm_down "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" || all_ok=false
    done
    if [[ "$all_ok" == "true" ]]; then
        echo "All vLLM workers confirmed down."
    else
        echo "Warning: one or more workers may still be running. Proceeding anyway."
    fi
  
}

# SSH to a worker and start vLLM inside a named tmux session.
# Args: host ip port rank model_dir
start_vllm_on_worker() {
    local host="$1" ip="$2" port="$3" rank="$4" model_dir="$5"
    local hf_env=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf_env="export HF_TOKEN='${HF_TOKEN}' HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'; "
    fi
    local cmd
    cmd=$(printf '%s' "$VLLM_START_CMD" \
        | sed "s|{model_dir}|${model_dir}|g" \
        | sed "s|{port}|${port}|g" \
        | sed "s|{rank}|${rank}|g" \
        | sed "s|{vllm_activate}|${VLLM_ACTIVATE}|g")
    echo "  [$host] starting vLLM in tmux '${VLLM_TMUX_SESSION}' (rank=$rank port=$port) ..."
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
        "${REMOTE_PATH} && ${hf_env}${cmd}" </dev/null
    sleep 2
    if ! ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
        "${REMOTE_PATH} && tmux has-session -t '${VLLM_TMUX_SESSION}'" </dev/null 2>/dev/null; then
        echo "Error: tmux session '${VLLM_TMUX_SESSION}' not found on $host — it exited immediately"
        echo "  Resolved command:"
        echo "    ${cmd}"
        echo "  vLLM log (/tmp/vllm_${rank}.log):"
        ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
            "tail -30 /tmp/vllm_${rank}.log 2>/dev/null || echo '  (no log found)'" </dev/null || true
        return 1
    fi
    echo "  [$host] vLLM started — attach with: ssh $host tmux attach -t ${VLLM_TMUX_SESSION}"
}

# Poll /health on a worker until it returns 200 — vLLM is ready.
# Uses globals: HEALTH_RETRIES HEALTH_INTERVAL
wait_for_vllm_up() {
    local host="$1" ip="$2" port="$3"
    local url="http://${ip}:${port}/health"
    local attempt=0
    echo "  [$host] waiting for vLLM to come up ($url) ..."
    while [[ $attempt -lt $HEALTH_RETRIES ]]; do
        if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
            echo "  [$host] vLLM is up ✓"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep "$HEALTH_INTERVAL"
    done
    echo "Error: vLLM on $host did not become healthy after $((HEALTH_RETRIES * HEALTH_INTERVAL))s"
    return 1
}

# Send a real completion request to confirm vLLM is serving correctly.
confirm_vllm_completion() {
    local host="$1" ip="$2" port="$3"
    local url="http://${ip}:${port}${COMPLETION_PATH}"
    local payload='{"prompt":"hello","max_tokens":5,"temperature":0.0}'
    echo "  [$host] sending completion confirmation request ..."
    local response
    response=$(curl -sf --max-time 15 -H "Content-Type: application/json" \
        -d "$payload" "$url") || {
        echo "Error: completion request failed on $host ($url)"
        return 1
    }
    local text
    text=$(echo "$response" | yq '.choices[0].text // "EMPTY"')
    echo "  [$host] completion ok — \"${text}\" ✓"
}

# ---------------------------------------------------------------------------
# Preflight: required commands
# ---------------------------------------------------------------------------

echo "Verifying required commands:"
require_cmd yq
require_cmd curl
require_cmd tmux
require_cmd ssh
require_cmd uv
echo "All required commands found."

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

if [[ ! -f "$GRPO_CONFIG" ]]; then
    echo "Error: GRPO config not found: $GRPO_CONFIG"
    exit 1
fi
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
    echo "Error: cluster config not found: $CLUSTER_CONFIG"
    exit 1
fi

VLLM_ENABLED=$(yq '.vllm' "$GRPO_CONFIG")
if [[ "$VLLM_ENABLED" != "true" ]]; then
    echo "Error: vllm must be enabled in GRPO config to use this launcher"
    exit 1
fi

NUM_WORKERS=$(yq '.num_workers' "$CLUSTER_CONFIG")
TOTAL_NUM_NODES=$(yq '.total_num_nodes' "$CLUSTER_CONFIG")
SERVER_HOST=$(yq '.server' "$CLUSTER_CONFIG")
VLLM_PORT=$(yq '.vllm_cluster.port' "$GRPO_CONFIG")
COMPLETION_PATH=$(yq '.vllm_cluster.completion_path' "$GRPO_CONFIG")
VLLM_ACTIVATE=$(yq '.weight_sync.vllm_activate' "$GRPO_CONFIG")
VLLM_START_CMD=$(yq '.weight_sync.vllm_start_cmd' "$GRPO_CONFIG")
HF_MODEL_NAME=$(yq '.dp.hf_model_name' "$MODEL_CONFIG")
HEALTH_RETRIES=$(yq '.weight_sync.health_retries // 30' "$GRPO_CONFIG")
HEALTH_INTERVAL=$(yq '.weight_sync.health_interval // 5' "$GRPO_CONFIG")

WORKER_HOSTS=()
WORKER_RANKS=()
WORKER_IPS=()
while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    host="${entry%%:*}"
    rank="${entry##*:}"
    ip=$(yq ".host_ip.${host}" "$CLUSTER_CONFIG")
    if [[ -z "$ip" || "$ip" == "null" ]]; then
        echo "Error: missing host_ip mapping for worker: $host"
        exit 1
    fi
    WORKER_HOSTS+=("$host")
    WORKER_RANKS+=("$rank")
    WORKER_IPS+=("$ip")
done < <(yq '.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$CLUSTER_CONFIG")

if [[ ${#WORKER_HOSTS[@]} -ne "$NUM_WORKERS" ]]; then
    echo "Error: num_workers ($NUM_WORKERS) does not match worker entries (${#WORKER_HOSTS[@]})"
    exit 1
fi

EXPECTED_TOTAL=$((NUM_WORKERS + 1))
if [[ "$TOTAL_NUM_NODES" -ne "$EXPECTED_TOTAL" ]]; then
    echo "Error: total_num_nodes ($TOTAL_NUM_NODES) must equal num_workers + 1 ($EXPECTED_TOTAL)"
    exit 1
fi

# ---------------------------------------------------------------------------
# --cleanup: kill all vLLM workers + confirm dead
# ---------------------------------------------------------------------------

if [[ "$CLEANUP" == "true" ]]; then
    reset_all_vllm_workers
    # Kill any background SSH log-tail processes started during launch
    if [[ -f "$GRPO_TAIL_PIDS_FILE" ]]; then
        echo "Stopping worker log-tail processes ..."
        while IFS= read -r _pid; do
            kill "$_pid" 2>/dev/null && echo "  killed tail pid $_pid" || true
        done < "$GRPO_TAIL_PIDS_FILE"
        rm -f "$GRPO_TAIL_PIDS_FILE"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch flow
# ---------------------------------------------------------------------------

echo ""
echo "Preflight checks"
echo "  Config:      $GRPO_CONFIG"
echo "  Server:      $SERVER_HOST"
echo "  Workers:     ${WORKER_HOSTS[*]}"
echo "  Port/path:   $VLLM_PORT$COMPLETION_PATH"
echo "  HF model:    $HF_MODEL_NAME"
echo ""

if [[ "$DRY_RUN" == "false" ]]; then

    # 1. SSH connectivity + tmux check on each worker
    echo "Checking SSH connectivity ..."
    for worker in "${WORKER_HOSTS[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$worker" "echo ok" >/dev/null 2>&1; then
            echo "Error: cannot SSH to worker: $worker"
            exit 1
        fi
        echo "  [$worker] SSH ok"
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$worker" "${REMOTE_PATH} && which tmux >/dev/null 2>&1"; then
            echo "  [$worker] tmux not found — installing ..."
            ssh -o ConnectTimeout=10 -o BatchMode=yes "$worker" \
                "sudo apt-get install -y tmux >/dev/null 2>&1 || brew install tmux >/dev/null 2>&1" || {
                echo "Error: could not install tmux on $worker — run: ssh $worker 'sudo apt-get install -y tmux'"
                exit 1
            }
            echo "  [$worker] tmux installed ✓"
        fi
    done

    # 2. Kill any stale vLLM instances + confirm they are down
    reset_all_vllm_workers

    # 3. Start fresh vLLM on all workers (all fire concurrently, then we poll each)
    echo ""
    echo "Starting vLLM on all workers ..."
    for i in "${!WORKER_HOSTS[@]}"; do
        start_vllm_on_worker "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" "${WORKER_RANKS[$i]}" "$HF_MODEL_NAME"
    done

    # 4. Wait for each worker's vLLM to pass /health
    echo ""
    echo "Waiting for vLLM workers to become healthy ..."
    for i in "${!WORKER_HOSTS[@]}"; do
        wait_for_vllm_up "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" || exit 1
    done

    # 5. Confirm with a real completion request ("hello") on each worker
    echo ""
    echo "Confirming vLLM completions ..."
    for i in "${!WORKER_HOSTS[@]}"; do
        confirm_vllm_completion "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" || exit 1
    done

else
    echo "Dry run: skipping SSH checks, vLLM reset, and endpoint health checks."
fi

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------

TRAIN_SCRIPT="$PROJECT_DIR/src/smolcluster/applications/reasoning/grpo/train_${TRAIN_TARGET}.py"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: training script not found: $TRAIN_SCRIPT"
    exit 1
fi

CLUSTER_LOG_DIR="$PROJECT_DIR/logging/cluster-logs"
mkdir -p "$CLUSTER_LOG_DIR"
SESSION_TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$CLUSTER_LOG_DIR/grpo-${TRAIN_TARGET}__${SERVER_HOST}__${SESSION_TS}.log"

# Stream each vLLM worker's log back to the server so the dashboard can display it.
# Each worker writes to /tmp/vllm_<rank>.log in its own tmux session; we tail that
# file over SSH and write to a local cluster-log file in the 3-part naming format
# (grpo-<target>__<worker>__<ts>.log) that the dashboard parser understands.
if [[ "$DRY_RUN" == "false" && ${#WORKER_HOSTS[@]} -gt 0 ]]; then
    rm -f "$GRPO_TAIL_PIDS_FILE"
    echo "Starting worker log streaming ..."
    for i in "${!WORKER_HOSTS[@]}"; do
        _w="${WORKER_HOSTS[$i]}"
        _r="${WORKER_RANKS[$i]}"
        _wlog="$CLUSTER_LOG_DIR/grpo-${TRAIN_TARGET}__${_w}__${SESSION_TS}.log"
        ssh -o ConnectTimeout=10 -o BatchMode=yes \
            -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
            "$_w" "${REMOTE_PATH}; tail -f -n +1 /tmp/vllm_${_r}.log 2>/dev/null" \
            >> "$_wlog" 2>/dev/null &
        echo $! >> "$GRPO_TAIL_PIDS_FILE"
        echo "  [${_w}] streaming vLLM log → $(basename "$_wlog")"
    done
fi

HF_ENV_SETUP=""
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ENV_SETUP="export HUGGING_FACE_HUB_TOKEN=\"${HF_TOKEN}\"; export HF_TOKEN=\"${HF_TOKEN}\"; "
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    HF_ENV_SETUP="export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; "
fi
HF_ENV_SETUP+="export HF_HUB_ENABLE_HF_TRANSFER=1; "

TRAIN_CMD="cd \"$PROJECT_DIR\" && ${HF_ENV_SETUP}uv run --group mlx python \"$TRAIN_SCRIPT\""

echo ""
echo "Launching GRPO training ($SERVER_HOST, target=$TRAIN_TARGET) ..."
echo "  Log file: $TRAIN_LOG"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run command: bash -lc '$TRAIN_CMD'"
else
    exec bash -lc "$TRAIN_CMD" 2>&1 | tee "$TRAIN_LOG"
fi
