#!/bin/bash

# GRPO evaluation launcher.
#
# passk mode:
#   Starts the exact same worker-side vLLM setup used by GRPO training,
#   then runs GSM8K pass@k evaluation through those workers.
#
# compare mode:
#   Runs local checkpoint comparison evaluation without standing up vLLM.
#
# Usage:
#   bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk
#   bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk --max-examples 25
#   bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh compare
#   bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh --cleanup

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
    source "$PROJECT_DIR/.env"
    set -u
fi

GRPO_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/reasoning/grpo/config.yaml"
CLUSTER_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
MODEL_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"
EVAL_SCRIPT="$PROJECT_DIR/src/smolcluster/applications/reasoning/grpo/evaluation/evaluate_gsm8k.py"
VLLM_TMUX_SESSION="vllm_worker"

MODE="passk"
DRY_RUN=false
CLEANUP=false
SKIP_SETUP=false
SFT_ADAPTERS_PATH=""
SFT_REMOTE_ADAPTER_PATH=""
EVAL_ARGS=()

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
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --sft-adapters)
            # Optional path argument; defaults to local SFT checkpoints
            if [[ $# -gt 1 && "$2" != --* ]]; then
                SFT_ADAPTERS_PATH="$2"
                shift 2
            else
                SFT_ADAPTERS_PATH="src/smolcluster/applications/sft/gsm8k/checkpoints"
                shift
            fi
            ;;
        --help|-h)
            echo "GRPO evaluation launcher"
            echo ""
            echo "Modes:"
            echo "  passk    Start training-style vLLM workers and run pass@k evaluation"
            echo "  compare  Run local checkpoint comparison (no worker setup)"
            echo ""
            echo "Options:"
            echo "  --dry-run          Print the resolved evaluation command"
            echo "  --cleanup          Kill vLLM workers on all configured GRPO workers"
            echo "  --skip-setup       Reuse already-running vLLM workers in passk mode"
            echo "  --sft-adapters [path]  Path to SFT LoRA adapters; defaults to local SFT folder"
            echo "                     Syncs adapters to workers and loads them in vLLM"
            echo ""
            echo "Examples:"
            echo "  bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk"
            echo "  bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk --max-examples 20"
            echo "  bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk --sft-adapters"
            echo "  bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk --sft-adapters checkpoints/sft_adapters"
            echo "  bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh compare"
            exit 0
            ;;
        passk|compare)
            MODE="$1"
            shift
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

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

has_eval_arg() {
    local needle="$1"
    local arg
    for arg in "${EVAL_ARGS[@]-}"; do
        if [[ "$arg" == "$needle" || "$arg" == "$needle="* ]]; then
            return 0
        fi
    done
    return 1
}

kill_vllm_on_worker() {
    local host="$1" port="$2"
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

reset_all_vllm_workers() {
    if [[ ${#WORKER_HOSTS[@]} -eq 0 ]]; then
        echo "  No workers configured, skipping vLLM reset."
        return 0
    fi

    echo "Killing vLLM on all workers ..."
    for i in "${!WORKER_HOSTS[@]}"; do
        kill_vllm_on_worker "${WORKER_HOSTS[$i]}" "$VLLM_PORT"
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

start_vllm_on_worker() {
    local host="$1" ip="$2" port="$3" rank="$4" model_dir="$5" adapter_path="${6:-}"
    local hf_env=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf_env="export HF_TOKEN='${HF_TOKEN}' HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'; "
    fi
    local cmd
    cmd=$(printf '%s' "$VLLM_START_CMD" \
        | sed "s|{model_dir}|${model_dir}|g" \
        | sed "s|{port}|${port}|g" \
        | sed "s|{rank}|${rank}|g" \
        | sed "s|{vllm_activate}|${VLLM_ACTIVATE}|g" \
        | sed "s|{adapter_path}|${adapter_path}|g")
    if [[ -n "$adapter_path" && "$cmd" != *"--adapter-path"* ]]; then
        local adapter_flag="--adapter-path ${adapter_path}"
        if [[ "$cmd" == *"2>&1"* ]]; then
            cmd="${cmd/2>&1/${adapter_flag} 2>&1}"
        else
            cmd="$cmd ${adapter_flag}"
        fi
    fi
    echo "  [$host] starting vLLM in tmux '${VLLM_TMUX_SESSION}' (rank=$rank port=$port) ..."
    if [[ -n "$adapter_path" ]]; then
        echo "  [$host] using adapter path: $adapter_path"
    fi
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
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
    echo "Error: cluster config not found: $CLUSTER_CONFIG"
    exit 1
fi
if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Error: model config not found: $MODEL_CONFIG"
    exit 1
fi


VLLM_ENABLED=$(yq '.vllm' "$GRPO_CONFIG")
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
CHECKPOINT_DIR_REL=$(yq '.weight_sync.checkpoint_dir // "checkpoints/grpo"' "$GRPO_CONFIG")
REMOTE_MODEL_DIR=$(yq '.weight_sync.remote_model_dir' "$GRPO_CONFIG")

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

if [[ "$CLEANUP" == "true" ]]; then
    reset_all_vllm_workers
    exit 0
fi

if [[ "$MODE" == "passk" && "$VLLM_ENABLED" != "true" ]]; then
    echo "Error: vllm must be enabled in GRPO config to use passk mode"
    exit 1
fi

if [[ "$MODE" == "passk" && "$SKIP_SETUP" == "false" ]]; then
    if [[ -n "$SFT_ADAPTERS_PATH" ]]; then
        if [[ -z "$REMOTE_MODEL_DIR" || "$REMOTE_MODEL_DIR" == "null" ]]; then
            echo "Error: weight_sync.remote_model_dir is missing in $GRPO_CONFIG"
            exit 1
        fi
        SFT_REMOTE_ADAPTER_PATH="${REMOTE_MODEL_DIR}_adapters"
    fi

    echo ""
    echo "Preflight checks"
    echo "  Config:      $GRPO_CONFIG"
    echo "  Server:      $SERVER_HOST"
    echo "  Workers:     ${WORKER_HOSTS[*]}"
    echo "  Port/path:   $VLLM_PORT$COMPLETION_PATH"
    echo "  HF model:    $HF_MODEL_NAME"
    if [[ -n "$SFT_ADAPTERS_PATH" ]]; then
        echo "  SFT adapters: $SFT_ADAPTERS_PATH (will be synced to workers)"
        echo "  Remote adapters: $SFT_REMOTE_ADAPTER_PATH"
        echo "  Adapter mode: enabled (vLLM will run with --adapter-path)"
    fi
    echo ""

    if [[ "$DRY_RUN" == "false" ]]; then
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
                    echo "Error: could not install tmux on $worker"
                    exit 1
                }
                echo "  [$worker] tmux installed ✓"
            fi
        done

        reset_all_vllm_workers

        if [[ -n "$SFT_ADAPTERS_PATH" ]]; then
            echo ""
            echo "Syncing SFT adapters to workers ..."
            if [[ ! -d "$PROJECT_DIR/$SFT_ADAPTERS_PATH" ]]; then
                echo "Error: SFT adapters directory not found: $PROJECT_DIR/$SFT_ADAPTERS_PATH"
                exit 1
            fi
            # Stage adapters in GRPO sync layout for worker_sync utility
            STAGED_ADAPTERS="$PROJECT_DIR/checkpoints/sft_eval_payload"
            mkdir -p "$STAGED_ADAPTERS/adapters"
            if [[ -f "$PROJECT_DIR/$SFT_ADAPTERS_PATH/adapters.safetensors" ]]; then
                cp "$PROJECT_DIR/$SFT_ADAPTERS_PATH/adapters.safetensors" "$STAGED_ADAPTERS/adapters/"
            fi
            if [[ -f "$PROJECT_DIR/$SFT_ADAPTERS_PATH/adapter_config.json" ]]; then
                cp "$PROJECT_DIR/$SFT_ADAPTERS_PATH/adapter_config.json" "$STAGED_ADAPTERS/adapters/"
            fi
            
            # Sync adapters to workers via Python utility
            set +u
            source "$VENV_ACTIVATE"
            set -u
            python3 << 'PYSYNC'
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path.cwd() / "src"))
from smolcluster.applications.reasoning.grpo.utils.worker_sync import sync_and_reload_workers

root = Path.cwd()
grpo_cfg = yaml.safe_load(open(root / 'src/smolcluster/configs/inference/reasoning/grpo/config.yaml'))
mdl_cfg = yaml.safe_load(open(root / 'src/smolcluster/configs/inference/model_config_inference.yaml'))
weights_dir = root / 'checkpoints/sft_eval_payload'

print(f"Syncing adapters from {weights_dir} to workers...")
sync_and_reload_workers(weights_dir, grpo_cfg, mdl_cfg)
print("Adapters synced and vLLM workers ready with SFT adapters!")
PYSYNC

        echo "SFT adapter sync complete. Worker-side vLLM uses adapter path: $SFT_REMOTE_ADAPTER_PATH"
        fi

        echo ""
        echo "Starting vLLM on all workers ..."
        for i in "${!WORKER_HOSTS[@]}"; do
            start_vllm_on_worker "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" "${WORKER_RANKS[$i]}" "$HF_MODEL_NAME" "$SFT_REMOTE_ADAPTER_PATH"
        done

        echo ""
        echo "Waiting for vLLM workers to become healthy ..."
        for i in "${!WORKER_HOSTS[@]}"; do
            wait_for_vllm_up "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" || exit 1
        done

        echo ""
        echo "Confirming vLLM completions ..."
        for i in "${!WORKER_HOSTS[@]}"; do
            confirm_vllm_completion "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "$VLLM_PORT" || exit 1
        done

        if [[ -n "$SFT_ADAPTERS_PATH" ]]; then
            echo ""
            echo "SFT mode active: evaluation will run against vLLM started with adapters"
            echo "  local adapter source: $SFT_ADAPTERS_PATH"
            echo "  remote --adapter-path: $SFT_REMOTE_ADAPTER_PATH"
        fi
    else
        echo "Dry run: skipping SSH checks, vLLM reset, and endpoint health checks."
    fi
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "Error: evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

if [[ "$MODE" == "passk" ]]; then
    if ! has_eval_arg "--model-path"; then
        EVAL_ARGS+=("--model-path" "$HF_MODEL_NAME")
    fi
    if ! has_eval_arg "--num-rollouts"; then
        EVAL_ARGS+=("--num-rollouts" "$(yq '.num_rollouts' "$GRPO_CONFIG")")
    fi
    if ! has_eval_arg "--use-vllm"; then
        EVAL_ARGS+=("--use-vllm")
    fi
else
    if ! has_eval_arg "--checkpoint-dir" && ! has_eval_arg "--step0" && ! has_eval_arg "--final"; then
        EVAL_ARGS+=("--checkpoint-dir" "$CHECKPOINT_DIR_REL")
    fi
fi

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/src/smolcluster/applications/reasoning/grpo/evaluation/eval-rollouts"

HF_ENV_SETUP=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ENV_SETUP+=("HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
    HF_ENV_SETUP+=("HF_TOKEN=${HF_TOKEN}")
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    HF_ENV_SETUP+=("HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
    HF_ENV_SETUP+=("HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
fi
HF_ENV_SETUP+=("HF_HUB_ENABLE_HF_TRANSFER=1")

echo ""
echo "Launching GRPO evaluation (mode=$MODE) ..."
echo "  Script: $EVAL_SCRIPT"
echo "  Args:   ${EVAL_ARGS[*]-}"
if [[ -n "$SFT_ADAPTERS_PATH" ]]; then
    echo "  SFT adapters enabled: YES"
    echo "  vLLM --adapter-path: $SFT_REMOTE_ADAPTER_PATH"
else
    echo "  SFT adapters enabled: NO"
fi

UV_RUN="env ${HF_ENV_SETUP[*]+${HF_ENV_SETUP[*]}} uv run --extra mlx --extra eval python"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'Dry run command: %s %q' "$UV_RUN" "$EVAL_SCRIPT"
    if [[ ${#EVAL_ARGS[@]} -gt 0 ]]; then
        printf ' %q' "${EVAL_ARGS[@]}"
    fi
    echo ""
    exit 0
fi

if [[ ${#EVAL_ARGS[@]} -gt 0 ]]; then
    env "${HF_ENV_SETUP[@]}" uv run --extra mlx --extra eval python "$EVAL_SCRIPT" "${EVAL_ARGS[@]}"
else
    env "${HF_ENV_SETUP[@]}" uv run --extra mlx --extra eval python "$EVAL_SCRIPT"
fi