#!/bin/bash

# Unified inference launcher for MP + DP
# Usage:
#   ./scripts/inference/launch_inference.sh --algorithm mp
#   ./scripts/inference/launch_inference.sh --algorithm syncps
#   ./scripts/inference/launch_inference.sh --algorithm classicdp
#   ./scripts/inference/launch_inference.sh --algorithm mp --dry-run
#   ./scripts/inference/launch_inference.sh --cleanup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"

ALGORITHM="mp"
DRY_RUN=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|-a)
            ALGORITHM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help|-h)
            echo "Unified inference launcher"
            echo ""
            echo "Options:"
            echo "  --algorithm, -a   mp | syncps | classicdp (default: mp)"
            echo "  --dry-run         Print commands without executing"
            echo "  --cleanup         Kill ALL smolcluster inference sessions and free ports (syncps + classicdp + mp)"
            echo "  --help, -h        Show help"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# --cleanup: kill every smolcluster inference session across all algorithms
# and free the shared API/frontend ports
# ---------------------------------------------------------------------------
do_cleanup() {
    echo ""
    echo "🧹 Cleaning up ALL smolcluster inference sessions (syncps / classicdp / mp)..."
    echo ""

    # Read ports from config (fall back to defaults if yq not available)
    local API_PORT=8000
    local FRONTEND_PORT=5050
    if [[ -f "$CONFIG_FILE" ]] && command -v yq >/dev/null 2>&1; then
        API_PORT=$(yq '.web_interface.api_port'    "$CONFIG_FILE" 2>/dev/null) || API_PORT=8000
        FRONTEND_PORT=$(yq '.web_interface.frontend_port' "$CONFIG_FILE" 2>/dev/null) || FRONTEND_PORT=5050
    fi

    # --- LOCAL tmux sessions ---
    echo "   Killing local tmux sessions..."
    local EXACT_SESSIONS=(
        syncps_api syncps_frontend
        classicdp_api classicdp_frontend
        mp_api mp_frontend
    )
    for session in "${EXACT_SESSIONS[@]}"; do
        if tmux has-session -t "$session" 2>/dev/null; then
            tmux kill-session -t "$session" 2>/dev/null
            echo "   [OK] Killed: $session"
        fi
    done

    # Pattern-matching local sessions (workers, tablets, server proxies)
    tmux ls 2>/dev/null | cut -d: -f1 \
        | grep -E '^(syncps_inf_|classicdp_inf_|mp_inference_|mp_tablet_proxy)' \
        | while IFS= read -r s; do
            tmux kill-session -t "$s" 2>/dev/null && echo "   [OK] Killed: $s"
        done

    # --- LOCAL ports ---
    echo ""
    echo "   Freeing local ports ($API_PORT, $FRONTEND_PORT)..."
    for PORT in "$API_PORT" "$FRONTEND_PORT"; do
        local PIDS
        PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null | sort -u)
        if [[ -n "$PIDS" ]]; then
            echo "$PIDS" | xargs kill 2>/dev/null || true
            sleep 1
            PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null | sort -u)
            [[ -n "$PIDS" ]] && { echo "$PIDS" | xargs kill -9 2>/dev/null || true; }
            echo "   [OK] Freed port $PORT"
        else
            echo "   [OK] Port $PORT already free"
        fi
    done

    # --- REMOTE tmux sessions via SSH ---
    echo ""
    echo "   Attempting remote cleanup on cluster nodes..."
    local SSH_HOSTS=()
    if [[ -f "$CONFIG_FILE" ]] && command -v yq >/dev/null 2>&1; then
        while IFS= read -r host; do
            [[ -n "$host" && "$host" != "null" ]] && SSH_HOSTS+=("$host")
        done < <(yq '.host_ip | keys | .[]' "$CONFIG_FILE" 2>/dev/null)
    fi

    if [[ ${#SSH_HOSTS[@]} -eq 0 ]]; then
        echo "   [SKIP] No SSH hosts found in config"
    else
        for host in "${SSH_HOSTS[@]}"; do
            # ipad connects inbound — we don't SSH to it
            [[ "$host" == "ipad" ]] && continue
            if ssh -o ConnectTimeout=3 -o BatchMode=yes "$host" \
                "tmux ls 2>/dev/null | cut -d: -f1 \
                 | grep -E '^(syncps_inf_|classicdp_inf_|mp_inference_|mp_tablet_proxy)' \
                 | xargs -r -I{} tmux kill-session -t {} 2>/dev/null; echo ok" \
                >/dev/null 2>&1; then
                echo "   [OK] Remote sessions cleaned on: $host"
            else
                echo "   [SKIP] $host unreachable or no matching sessions"
            fi
        done
    fi

    echo ""
    echo "✅ All smolcluster inference sessions terminated and ports freed."
    echo ""
}

if [[ "$CLEANUP" == "true" ]]; then
    do_cleanup
    exit 0
fi

case "$ALGORITHM" in
    mp)
        TARGET_SCRIPT="$SCRIPT_DIR/model_parallelism/launch_mp_inference.sh"
        ;;
    syncps)
        TARGET_SCRIPT="$SCRIPT_DIR/data_parallelism/launch_syncps_inference.sh"
        ;;
    classicdp)
        TARGET_SCRIPT="$SCRIPT_DIR/data_parallelism/launch_classicdp_inference.sh"
        ;;
    *)
        echo "❌ Unsupported algorithm: $ALGORITHM"
        echo "Supported: mp, syncps, classicdp"
        exit 1
        ;;
esac

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "❌ Missing target script: $TARGET_SCRIPT"
    exit 1
fi

if [[ "$DRY_RUN" != "true" ]]; then
    do_cleanup
fi

echo "🚀 Launching inference with algorithm: $ALGORITHM"

if [[ "$DRY_RUN" == "true" ]]; then
    bash "$TARGET_SCRIPT" --dry-run
else
    bash "$TARGET_SCRIPT"
fi
