#!/bin/bash
# Launch lm-evaluation-harness for SFT adapters (fuse-first by default).
#
# Default flow:
#   1) Fuse local SFT adapters into a standalone model.
#   2) Run lm_eval on single-turn instruction-following benchmarks.
#
# Pass --eval-model to skip adapters entirely and evaluate any HF model or
# local path directly.
#
# Usage:
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --tasks gsm8k_cot_zeroshot,ifeval
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --no-fuse
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --dry-run
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --eval-model meta-llama/Llama-3.2-1B-Instruct
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --eval-model /path/to/local/model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"
FUSE_SCRIPT="$SCRIPT_DIR/fuse_adapters.sh"

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
    echo "Error: cannot find project root (pyproject.toml / .git)"
    exit 1
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set +u
    source "$PROJECT_DIR/.env"
    set -u
fi

MODEL_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: .venv not found. Run 'uv sync' inside $PROJECT_DIR first."
    exit 1
fi

if ! command -v yq >/dev/null 2>&1; then
    echo "Error: yq is required. Install with: brew install yq"
    exit 1
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Error: model config not found: $MODEL_CONFIG"
    exit 1
fi

HF_MODEL_NAME=$(yq '.dp.hf_model_name' "$MODEL_CONFIG")
ADAPTER_PATH="$SFT_DIR/checkpoints"
FUSED_MODEL_PATH="$PROJECT_DIR/src/smolcluster/applications/sft/gsm8k/checkpoints/sft_final_fused"
TASKS="arc_challenge"
NUM_FEWSHOT=0

BATCH_SIZE="8"
DEVICE="mps"  # change to "cuda" if using an NVIDIA GPU; "cpu" also works but is slow
LIMIT=""
OUTPUT_PATH=""
DRY_RUN=false
FUSE_FIRST=true
DIRECT_EVAL_MODEL=""   # set via --eval-model; bypasses all adapter logic
EXTRA_ARGS=()
HAS_EXTRA_ARGS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --fused-model-path)
            FUSED_MODEL_PATH="$2"
            shift 2
            ;;
        --model)
            HF_MODEL_NAME="$2"
            shift 2
            ;;
        --num-fewshot)
            NUM_FEWSHOT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --eval-model)
            DIRECT_EVAL_MODEL="$2"
            shift 2
            ;;
        --no-fuse)
            FUSE_FIRST=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh [options]"
            echo ""
            echo "Options:"
            echo "  --tasks <csv>            lm_eval tasks (default: gsm8k_cot_zeroshot,ifeval,mmlu,arc_challenge,hellaswag)"
            echo "  --adapter-path <path>    LoRA adapter dir (default: local SFT checkpoints)"
            echo "  --fused-model-path <p>   Fused model output/input path"
            echo "  --model <hf_model>       Base HF model name (from model config by default)"
            echo "  --num-fewshot <n>        Few-shot examples per task (default: 0)"
            echo "  --batch-size <value>     lm_eval batch size (default: auto)"
            echo "  --device <dev>           lm_eval device (default: mps on macOS)"
            echo "  --limit <n|f>            lm_eval --limit for quick runs"
            echo "  --output-path <path>     lm_eval output path"
            echo "  --eval-model <hf|path>   Evaluate any HF model or local path directly (skips all adapter logic)"
            echo "  --no-fuse                Evaluate with base+peft adapter (skip fuse step)"
            echo "  --dry-run                Print commands only"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            HAS_EXTRA_ARGS=true
            shift
            ;;
    esac
done

if [[ "$ADAPTER_PATH" != /* ]]; then
    ADAPTER_PATH="$PROJECT_DIR/$ADAPTER_PATH"
fi
if [[ "$FUSED_MODEL_PATH" != /* ]]; then
    FUSED_MODEL_PATH="$PROJECT_DIR/$FUSED_MODEL_PATH"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_PATH="$PROJECT_DIR/src/smolcluster/applications/sft/gsm8k/eval-results/lm_eval_${ts}.json"
elif [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$PROJECT_DIR/$OUTPUT_PATH"
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"

set +u
source "$VENV_ACTIVATE"
set -u

cd "$PROJECT_DIR"

if ! python -c "import lm_eval" >/dev/null 2>&1; then
    echo "lm_eval not found in .venv; installing lm-eval..."
    uv pip install lm-eval
    uv pip install accelerate  # ensure accelerate is installed for GPU support
fi

# Install benchmark dependencies task-by-task from TASKS.
IFS=',' read -r -a SELECTED_TASKS <<< "$TASKS"
for task in "${SELECTED_TASKS[@]}"; do
    task_trimmed="$(echo "$task" | xargs)"
    if [[ -z "$task_trimmed" ]]; then
        continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Dry-run benchmark dependency install: uv pip install \"lm-eval[$task_trimmed]\""
    else
        echo "Ensuring lm-eval benchmark dependency: $task_trimmed"
        if ! uv pip install "lm-eval[$task_trimmed]"; then
            echo "Warning: lm-eval extra '$task_trimmed' may not exist; continuing."
        fi
    fi
done

EVAL_MODEL_PATH=""
MODEL_ARGS=""

if [[ -n "$DIRECT_EVAL_MODEL" ]]; then
    echo "SFT adapters enabled: NO (direct model mode)"
    echo "  model: $DIRECT_EVAL_MODEL"
    EVAL_MODEL_PATH="$DIRECT_EVAL_MODEL"
    MODEL_ARGS="pretrained=${EVAL_MODEL_PATH},trust_remote_code=True"
elif [[ "$FUSE_FIRST" == "true" ]]; then
    if [[ ! -f "$FUSE_SCRIPT" ]]; then
        echo "Error: fuse script not found: $FUSE_SCRIPT"
        exit 1
    fi

    echo "SFT adapters enabled: YES (fuse-first mode)"
    echo "  adapter path: $ADAPTER_PATH"
    echo "  fused output: $FUSED_MODEL_PATH"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo "Dry-run fuse command:"
        printf '  bash %q --model %q --adapter-path %q --save-path %q\n' "$FUSE_SCRIPT" "$HF_MODEL_NAME" "$ADAPTER_PATH" "$FUSED_MODEL_PATH"
    else
        bash "$FUSE_SCRIPT" --model "$HF_MODEL_NAME" --adapter-path "$ADAPTER_PATH" --save-path "$FUSED_MODEL_PATH"
    fi

    EVAL_MODEL_PATH="$FUSED_MODEL_PATH"
    MODEL_ARGS="pretrained=${EVAL_MODEL_PATH},trust_remote_code=True"
else
    echo "SFT adapters enabled: YES (direct PEFT mode, no fuse)"
    echo "  base model: $HF_MODEL_NAME"
    echo "  adapter path: $ADAPTER_PATH"

    EVAL_MODEL_PATH="$HF_MODEL_NAME"
    MODEL_ARGS="pretrained=${EVAL_MODEL_PATH},peft=${ADAPTER_PATH},trust_remote_code=True"
fi  # end adapter/direct-model branch

LM_EVAL_CMD=(
    python -m lm_eval
    --model hf
    --model_args "$MODEL_ARGS"
    --tasks "$TASKS"
    --num_fewshot "$NUM_FEWSHOT"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
    --output_path "$OUTPUT_PATH"
)

if [[ -n "$LIMIT" ]]; then
    LM_EVAL_CMD+=(--limit "$LIMIT")
fi
if [[ "$HAS_EXTRA_ARGS" == "true" ]]; then
    LM_EVAL_CMD+=("${EXTRA_ARGS[@]}")
fi

echo ""
echo "Launching lm_eval..."
echo "  eval model source: $EVAL_MODEL_PATH"
echo "  tasks: $TASKS"
echo "  output: $OUTPUT_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry-run lm_eval command:"
    printf '  %q' "${LM_EVAL_CMD[@]}"
    echo ""
    exit 0
fi

"${LM_EVAL_CMD[@]}"

echo ""
echo "lm_eval complete. Results saved to: $OUTPUT_PATH"
