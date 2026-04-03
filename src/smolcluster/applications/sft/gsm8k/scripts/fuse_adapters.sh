#!/bin/bash
# Fuse SFT LoRA adapters into a standalone final model.
#
# Usage:
#   bash src/smolcluster/applications/sft/gsm8k/scripts/fuse_adapters.sh
#   bash src/smolcluster/applications/sft/gsm8k/scripts/fuse_adapters.sh --adapter-path src/smolcluster/applications/sft/gsm8k/checkpoints --save-path checkpoints/sft_final_fused
#   bash src/smolcluster/applications/sft/gsm8k/scripts/fuse_adapters.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"

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
SAVE_PATH="checkpoints/sft_final_fused"
DRY_RUN=false
EXTRA_ARGS=()
HAS_EXTRA_ARGS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --model)
            HF_MODEL_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash src/smolcluster/applications/sft/gsm8k/scripts/fuse_adapters.sh [--adapter-path <path>] [--save-path <path>] [--model <hf_model>] [--dry-run] [extra mlx_lm fuse flags...]"
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
if [[ "$SAVE_PATH" != /* ]]; then
    SAVE_PATH="$PROJECT_DIR/$SAVE_PATH"
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "Error: adapter path not found: $ADAPTER_PATH"
    exit 1
fi
if [[ ! -f "$ADAPTER_PATH/adapters.safetensors" ]]; then
    echo "Error: missing adapters.safetensors in $ADAPTER_PATH"
    exit 1
fi
if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    echo "Error: missing adapter_config.json in $ADAPTER_PATH"
    exit 1
fi

mkdir -p "$SAVE_PATH"

echo "Base model   : $HF_MODEL_NAME"
echo "Adapter path : $ADAPTER_PATH"
echo "Save path    : $SAVE_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry-run - resolved command:"
    printf '  python -m mlx_lm fuse --model %q --adapter-path %q --save-path %q' "$HF_MODEL_NAME" "$ADAPTER_PATH" "$SAVE_PATH"
    if [[ "$HAS_EXTRA_ARGS" == "true" ]]; then
        printf ' %q' "${EXTRA_ARGS[@]}"
    fi
    echo ""
    exit 0
fi

set +u
source "$VENV_ACTIVATE"
set -u

cd "$PROJECT_DIR"
echo "Installing project dependencies..."
uv pip install -e . -q

FUSE_CMD=(
    env HF_HUB_ENABLE_HF_TRANSFER=1 python -m mlx_lm fuse
    --model "$HF_MODEL_NAME"
    --adapter-path "$ADAPTER_PATH"
    --save-path "$SAVE_PATH"
)
if [[ "$HAS_EXTRA_ARGS" == "true" ]]; then
    FUSE_CMD+=("${EXTRA_ARGS[@]}")
fi

"${FUSE_CMD[@]}"

echo ""
echo "Fusion complete. Final fused model written to: $SAVE_PATH"
echo "Use this fused model path for standalone inference/eval if needed."
