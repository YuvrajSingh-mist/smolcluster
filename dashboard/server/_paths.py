"""File-system path constants used across the dashboard server."""
from pathlib import Path

FRONTEND_DIR  = Path(__file__).resolve().parents[2] / "frontend"

# /tmp sentinel files written by training/inference processes
METRICS_FILE    = Path("/tmp/smolcluster_metrics.json")
INFERENCE_FILE  = Path("/tmp/smolcluster_inference.json")
TOKEN_PING      = Path("/tmp/smolcluster_token_ping")
LAST_TOKEN      = Path("/tmp/smolcluster_last_token")
TOKEN_INTERVAL  = Path("/tmp/smolcluster_token_interval_ms")
GRAD_PING       = Path("/tmp/smolcluster_grad_ping")
GRAD_INTERVAL   = Path("/tmp/smolcluster_grad_interval_ms")

CLUSTER_LOG_DIR = Path(__file__).resolve().parents[3] / "logging" / "cluster-logs"

# Repository paths (one level above the dashboard/ package)
_REPO_ROOT = Path(__file__).resolve().parents[3]

INFER_CONFIG_FILE = (
    _REPO_ROOT / "src" / "smolcluster" / "configs" / "inference" / "cluster_config_inference.yaml"
)
INFER_SCRIPT_FILE     = _REPO_ROOT / "scripts" / "inference" / "launch_inference.sh"
GRPO_TRAIN_SCRIPT_FILE = (
    _REPO_ROOT / "src" / "smolcluster" / "applications"
    / "reasoning" / "grpo" / "scripts" / "launch_grpo_train.sh"
)

TRAIN_CONFIGS_DIR = str(_REPO_ROOT / "src" / "smolcluster" / "configs")
TRAIN_SCRIPTS_DIR = str(_REPO_ROOT / "scripts" / "training")
