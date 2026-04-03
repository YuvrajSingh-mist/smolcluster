#!/bin/bash
# Launch MLX-LM LoRA SFT on GSM8K.
#
# Usage:
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh                    # full run
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --skip-data        # skip data prep (data/ already built)
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --iters 500        # override training iterations
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --dry-run          # show resolved command only
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --test             # run test-set eval after training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"  # …/sft/gsm8k

find_project_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/pyproject.toml" ]] || [[ -d "$dir/.git" ]]; then
            echo "$dir"; return 0
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
    set +u; source "$PROJECT_DIR/.env"; set -u
fi

MODEL_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"
LORA_CONFIG="$SFT_DIR/lora_config.yaml"
PREPARE_SCRIPT="$SFT_DIR/prepare_data.py"
DATA_DIR="$SFT_DIR/data"

VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: .venv not found. Run 'uv sync' inside $PROJECT_DIR first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_DATA=false
DRY_RUN=false
RUN_TEST=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data) SKIP_DATA=true;  shift ;;
        --dry-run)   DRY_RUN=true;    shift ;;
        --test)      RUN_TEST=true;   shift ;;
        --help|-h)
            echo "Usage: bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh [--skip-data] [--dry-run] [--test] [extra mlx_lm lora flags...]"
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve model name from model config (requires yq)
# ---------------------------------------------------------------------------
if ! command -v yq >/dev/null 2>&1; then
    echo "Error: yq is required. Install with: brew install yq"
    exit 1
fi
HF_MODEL_NAME=$(yq '.dp.hf_model_name' "$MODEL_CONFIG")
echo "Model : $HF_MODEL_NAME"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
set +u; source "$VENV_ACTIVATE"; set -u

cd "$PROJECT_DIR"
echo "Installing project dependencies..."
uv pip install -e . -q

if [[ "$SKIP_DATA" == "false" ]]; then
    echo ""
    echo "Preparing GSM8K data..."
    python "$PREPARE_SCRIPT" --model-name "$HF_MODEL_NAME" --out-dir "$DATA_DIR"
else
    echo "Skipping data preparation (--skip-data)"
    if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
        echo "Error: $DATA_DIR/train.jsonl not found. Run without --skip-data first."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Build mlx_lm lora command
# ---------------------------------------------------------------------------
LORA_ARGS=(
    "--train"
    "--model"   "$HF_MODEL_NAME"
    "--data"    "$DATA_DIR"
    "--config"  "$LORA_CONFIG"
)

if [[ "$RUN_TEST" == "true" ]]; then
    LORA_ARGS+=("--test")
fi

# Append any extra flags (e.g. --iters 500) passed through to this script
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    LORA_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry-run — resolved command:"
    printf '  python -m mlx_lm lora'
    printf ' %q' "${LORA_ARGS[@]}"
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Run SFT
# ---------------------------------------------------------------------------
echo ""
echo "Launching SFT fine-tuning..."
echo "  Config  : $LORA_CONFIG"
echo "  Data    : $DATA_DIR"
echo "  Adapters: $SFT_DIR/checkpoints"
echo ""

HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ENV+=("HF_TOKEN=${HF_TOKEN}" "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi
HF_ENV+=("HF_HUB_ENABLE_HF_TRANSFER=1")

env "${HF_ENV[@]}" python -m mlx_lm lora "${LORA_ARGS[@]}"

echo ""
echo "SFT complete. Adapters saved to: $SFT_DIR/checkpoints"
echo ""
echo "To fuse adapters into a standalone model:"
echo "  bash src/smolcluster/applications/sft/gsm8k/scripts/fuse_adapters.sh"
