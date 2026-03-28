"""
Utilities for syncing updated policy weights to vLLM rollout workers.

GRPO requires the vLLM workers generating rollouts to use the current policy.
After every `weight_sync.save_steps` training steps this module:

  1. Saves the policy model weights as a safetensors file locally.
  2. SCPs the weights file to each worker's remote model directory,
     overwriting the existing weights in-place.
  3. Kills the running vLLM process on each worker via SSH.
  4. Restarts vLLM on each worker with the updated model directory.
  5. Polls each worker's /health endpoint until it is ready.

Required config.yaml keys (under `weight_sync`):
  save_steps          int   – sync every N steps (0 = disabled)
  checkpoint_dir      str   – local path relative to project root, e.g. "checkpoints/grpo"
  remote_model_dir    str   – absolute path on workers that contains the full model
                              (config.json + tokenizer + weights), e.g. "/home/ubuntu/grpo_model"

From cluster_config_inference.yaml:
  host_ip             dict  – hostname → IP mapping, e.g. {mini1: "10.10.0.1", ...}
  workers.regular[].user str – per-worker SSH user, e.g. yuvrajsingh2
  remote_weights_filename str – filename inside remote_model_dir to overwrite, default "model.safetensors"
  vllm_start_cmd      str   – shell command to (re)start vLLM; {model_dir}, {port}, {rank},
                              {vllm_activate} are substituted.  Default: tmux-based launch.
  health_retries      int   – number of health-check attempts before giving up (default 30)
  health_interval     int   – seconds between health checks (default 5)
"""

import logging
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import requests
import yaml
from mlx.utils import tree_flatten
import os

logger = logging.getLogger(__name__)

_module_dir = Path(__file__).parent
_smolcluster_root = _module_dir.parents[2]
# Project root: two levels above the smolcluster source package
_project_root = _smolcluster_root.parent.parent

_cluster_config_path = _smolcluster_root / "configs" / "inference" / "cluster_config_inference.yaml"
with open(_cluster_config_path) as _f:
    _cluster_cfg: Dict[str, Any] = yaml.safe_load(_f)

_DEFAULT_VLLM_START_CMD = (
    "tmux kill-session -t vllm_worker 2>/dev/null || true; "
    "tmux new-session -d -s vllm_worker "
    "\"bash -c 'source {vllm_activate} && vllm serve {model_dir} --port {port} "
    "--trust-remote-code --enable-prefix-caching 2>&1 | tee /tmp/vllm_{rank}.log; exec bash'\""
)


# ---------------------------------------------------------------------------
# Weight saving
# ---------------------------------------------------------------------------

def save_policy_weights(model: Any, checkpoint_dir: str, step: int) -> Path:
    """
    Save policy weights (or LoRA adapters for quantized models) as safetensors.

    For quantized models with LoRA adapters (detected via the presence of a
    ``fuse`` method on any module), saves only the trainable adapter weights
    plus an ``adapter_config.json`` to ``step_dir/adapters/``.  Workers then
    load the unchanged 4-bit base model and overlay the adapters via
    ``--adapter-path``.

    For non-quantized models, saves the full ``model.safetensors`` (existing
    behaviour).

    Args:
        model:          The MLX policy model (not the reference model).
        checkpoint_dir: Path relative to the project root where checkpoints are stored.
        step:           Current global training step, used to name the subdirectory.

    Returns:
        The step-level directory that was written, e.g.
        ``<project_root>/checkpoints/grpo/step_10/``.
    """
    import json

    step_dir = _project_root / checkpoint_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    lora_modules = [(n, m) for n, m in model.named_modules() if hasattr(m, "fuse")]
    if lora_modules:
        # Save only LoRA adapter weights — workers keep the original 4-bit base model
        adapter_dir = step_dir / "adapters"
        adapter_dir.mkdir(exist_ok=True)
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(list(adapter_weights.values()))
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), adapter_weights)

        # Write adapter_config.json — infer rank and scale from the first LoRA module
        _, first_lora = lora_modules[0]
        rank = int(first_lora.lora_a.shape[0])
        scale = float(first_lora.scale)
        adapter_cfg = {
            "lora_parameters": {
                "rank": rank,
                "alpha": scale * rank,
                "dropout": 0.0,
                "scale": scale,
            }
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2))
        logger.info("[weight_sync] Step %d — LoRA adapters saved to %s", step, adapter_dir)
        return step_dir

    # Non-quantized model: full weight save (existing behaviour)
    weights_path = step_dir / "model.safetensors"
    flat_weights = dict(tree_flatten(model.parameters()))
    # Force evaluation so all pending MLX operations complete before we write.
    mx.eval(list(flat_weights.values()))
    mx.save_safetensors(str(weights_path), flat_weights)
    logger.info("[weight_sync] Step %d — weights saved to %s", step, weights_path)
    return step_dir


def _get_local_model_dir(model_name: str) -> Path:
    """
    Locate the local HF cache directory for a model.
    Assumes the model is already cached locally at ~/.cache/huggingface/hub/models--{org}--{model}.
    Returns the latest snapshot directory (containing symlinks to actual model files).
    """
    
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    # Model repo dirs are stored as models--{org}--{model_name} (with / converted to --)
    cache_name = model_name.replace("/", "--")
    base_model_dir = hf_home / "hub" / f"models--{cache_name}"
    
    snapshots_dir = base_model_dir / "snapshots"
    if not snapshots_dir.exists():
        raise RuntimeError(f"No snapshots directory found at {snapshots_dir}")
    
    # Get the latest snapshot (there should be only one for a given model)
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise RuntimeError(f"No snapshot subdirectories found in {snapshots_dir}")
    
    latest_snapshot = snapshots[0]  # Assume single snapshot; if multiple, get most recent
    return latest_snapshot


def _scp_model_files(
    hostname: str,
    local_model_dir: Path,
    remote_model_dir: str,
) -> None:
    """
    Rsync all non-weight model files (config, tokenizer, vocab, etc.) to the remote
    worker. Uses --exclude to skip weight files so only architecture/tokenizer files
    are transferred. This ensures vLLM gets every file it needs regardless of model
    architecture (vocab.json, merges.txt, added_tokens.json, etc.).
    """
    rsync_cmd = [
        "rsync", "-azL",  # -L follows symlinks (HF cache uses symlinks to blobs)
        "--exclude=*.safetensors",
        "--exclude=*.bin",
        "--exclude=*.pt",
        "--exclude=*.index.json",
        f"{local_model_dir}/",
        f"{hostname}:{remote_model_dir}/",
    ]
    logger.info("[weight_sync] rsync model files → %s:%s", hostname, remote_model_dir)
    try:
        result = subprocess.run(rsync_cmd, timeout=180, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "[weight_sync] rsync model files to %s failed (rc=%d): %s",
                hostname, result.returncode, result.stderr.strip(),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("[weight_sync] rsync model files failed: %s, continuing ...", e)


# ---------------------------------------------------------------------------
# SSH / SCP helpers
# ---------------------------------------------------------------------------

def _run_ssh(
    hostname: str,
    cmd: str,
    timeout: int = 60,
    use_login_shell: bool = False,
) -> subprocess.CompletedProcess:
    if use_login_shell:
        cmd = f"zsh -lc {shlex.quote(cmd)}"
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        hostname,
        cmd,
    ]
    logger.debug("[weight_sync] SSH %s: %s", hostname, cmd)
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def _scp_file(
    local_path: Path,
    hostname: str,
    remote_path: str,
    timeout: int = 180,
) -> None:
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        str(local_path),
        f"{hostname}:{remote_path}",
    ]
    logger.info(
        "[weight_sync] SCP %s → %s:%s",
        local_path.name, hostname, remote_path,
    )
    # Don't use capture_output so password prompts can appear interactively
    result = subprocess.run(scp_cmd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"SCP to {hostname}:{remote_path} failed (rc={result.returncode})"
        )


def _resolve_remote_path(hostname: str, path: str) -> str:
    """Expand ~ and env vars on the remote host and return an absolute path."""
    py = f"import os; print(os.path.expandvars(os.path.expanduser({path!r})))"
    result = _run_ssh(hostname, f"python3 -c {shlex.quote(py)}", use_login_shell=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not resolve remote path '{path}' on {hostname}: {result.stderr.strip()}"
        )
    resolved = result.stdout.strip()
    if not resolved:
        raise RuntimeError(f"Could not resolve remote path '{path}' on {hostname}: empty output")
    return resolved


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _wait_for_health(health_url: str, retries: int, interval: int) -> None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                logger.info("[weight_sync] Worker healthy: %s", health_url)
                return
        except Exception:
            pass
        logger.info(
            "[weight_sync] Health check %d/%d for %s ...", attempt, retries, health_url
        )
        time.sleep(interval)
    raise RuntimeError(
        f"Worker at {health_url} did not become healthy after {retries} attempts "
        f"({retries * interval}s total)"
    )


# ---------------------------------------------------------------------------
# Per-worker sync
# ---------------------------------------------------------------------------

def _sync_single_worker(
    rank: int,
    hostname: str,
    host_ip: str,
    local_weights_dir: Path,
    sync_cfg: Dict[str, Any],
    vllm_cluster: Dict[str, Any],
    local_model_dir: Path = None,
) -> None:
    remote_model_dir = _resolve_remote_path(hostname, sync_cfg["remote_model_dir"])
    remote_weights_filename = sync_cfg.get("remote_weights_filename", "model.safetensors")
    port = int(vllm_cluster.get("port", 8000))

    vllm_activate = sync_cfg.get("vllm_activate", "~/.venv-vllm-metal/bin/activate")
    vllm_activate = _resolve_remote_path(hostname, vllm_activate)

    # Detect LoRA mode: adapters were saved instead of a full weights file
    adapter_dir = local_weights_dir / "adapters"
    is_lora = adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists()
    remote_adapter_dir = remote_model_dir + "_adapters"

    vllm_start_cmd = sync_cfg.get("vllm_start_cmd", _DEFAULT_VLLM_START_CMD).format(
        model_dir=remote_model_dir,
        port=port,
        rank=rank,
        vllm_activate=vllm_activate,
    )
    if is_lora:
        # Inject --adapter-path before the output redirect so it lands inside
        # the bash -c string regardless of the template shape.
        adapter_flag = f"--adapter-path {shlex.quote(remote_adapter_dir)}"
        if "2>&1" in vllm_start_cmd:
            vllm_start_cmd = vllm_start_cmd.replace("2>&1", f"{adapter_flag} 2>&1", 1)
        else:
            vllm_start_cmd += f" {adapter_flag}"

    local_weights_file = local_weights_dir / "model.safetensors"
    remote_weights_path = f"{remote_model_dir}/{remote_weights_filename}"

    # 1. Ensure the destination directory exists on the worker.
    mkdir_result = _run_ssh(hostname, f"mkdir -p {shlex.quote(remote_model_dir)}")
    if mkdir_result.returncode != 0:
        raise RuntimeError(
            f"Could not create remote model dir on worker {rank} ({hostname}): "
            f"{mkdir_result.stderr.strip()}"
        )

    # 2. Sync model config and tokenizer files (required by vLLM) if available locally.
    if local_model_dir and local_model_dir.exists():
        logger.info("[weight_sync] worker %d (%s): syncing model files ...", rank, hostname)
        _scp_model_files(hostname, local_model_dir, remote_model_dir)

    # 3. Clean stale weight shards on remote before uploading new weights.
    # Old shard files (model-00001-of-NNNNN.safetensors, model.safetensors.index.json)
    # left from a previous model setup would cause vLLM to load wrong/mixed weights.
    clean_cmd = (
        f"rm -f {shlex.quote(remote_model_dir)}/*.safetensors "
        f"{shlex.quote(remote_model_dir)}/*.bin "
        f"{shlex.quote(remote_model_dir)}/*.pt "
        f"{shlex.quote(remote_model_dir)}/model.safetensors.index.json"
    )
    _run_ssh(hostname, clean_cmd)
    logger.info("[weight_sync] worker %d (%s): stale remote weights cleaned", rank, hostname)

    # 4. Copy weights or LoRA adapters to the worker.
    if is_lora:
        _run_ssh(hostname, f"mkdir -p {shlex.quote(remote_adapter_dir)}")
        for fname in ["adapters.safetensors", "adapter_config.json"]:
            _scp_file(adapter_dir / fname, hostname, f"{remote_adapter_dir}/{fname}")
        logger.info("[weight_sync] worker %d (%s): LoRA adapters uploaded", rank, hostname)
    else:
        _scp_file(local_weights_file, hostname, remote_weights_path)
        logger.info("[weight_sync] worker %d (%s): weights uploaded", rank, hostname)

    # 5. Kill the running vLLM tmux session and any stale process on the worker.
    kill_result = _run_ssh(
        hostname,
        "tmux kill-session -t vllm_worker 2>/dev/null || true; pkill -f 'vllm serve' 2>/dev/null || true",
    )
    logger.info(
        "[weight_sync] worker %d (%s): pkill vllm → rc=%d stderr=%s",
        rank, hostname, kill_result.returncode, kill_result.stderr.strip() or "-",
    )
    # Brief pause to let the port free up before restarting.
    time.sleep(2)

    # 6. Restart vLLM.
    start_result = _run_ssh(hostname, vllm_start_cmd, timeout=30, use_login_shell=True)
    if start_result.returncode != 0:
        raise RuntimeError(
            f"Could not start vLLM on worker {rank} ({hostname}): "
            f"{start_result.stderr.strip()}"
        )
    logger.info("[weight_sync] worker %d (%s): vLLM restart command sent", rank, hostname)

    # 7. Poll /health until the server is ready.
    health_url = f"http://{host_ip}:{port}/health"
    retries = int(sync_cfg.get("health_retries", 30))
    interval = int(sync_cfg.get("health_interval", 5))
    _wait_for_health(health_url, retries, interval)
    logger.info("[weight_sync] worker %d (%s): ready", rank, hostname)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_and_reload_workers(
    local_weights_dir: Path,
    grpo_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> None:
    """
    Distribute the saved policy weights to all vLLM workers, kill their running
    vLLM processes, restart them, and wait until each is healthy.

    Workers are processed in parallel; if any worker fails the exception is
    collected and re-raised after all threads finish.

    Args:
        local_weights_dir: The step directory returned by ``save_policy_weights``.
        grpo_config:        The loaded GRPO configuration dict (contains ``vllm_cluster``
                            and ``weight_sync`` sections).
        model_config:       The loaded model configuration dict (contains model name for HF cache lookup).
    """
    sync_cfg = grpo_config.get("weight_sync", {})
    vllm_cluster = grpo_config["vllm_cluster"]
    # host_ip, workers, and per-node user all come from cluster_config_inference.yaml
    host_ip_map = _cluster_cfg["host_ip"]
    num_workers = int(_cluster_cfg["num_workers"])
    workers = _cluster_cfg["workers"]["regular"][:num_workers]

    # Get the local model directory from HF cache
    model_name = model_config["dp"]["hf_model_name"]
    try:
        local_model_dir = _get_local_model_dir(model_name)
        logger.info("[weight_sync] Found local model at %s", local_model_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("[weight_sync] Could not find local model dir: %s, skipping model file sync", e)
        local_model_dir = None

    logger.info(
        "[weight_sync] Distributing weights from %s to %d worker(s) ...",
        local_weights_dir,
        len(workers),
    )

    errors: Dict[int, Exception] = {}
    errors_lock = threading.Lock()

    def _task(worker: Dict[str, Any]) -> None:
        rank = worker["rank"]
        hostname = worker["hostname"]
        ip = host_ip_map[hostname]
        try:
            _sync_single_worker(rank, hostname, ip, local_weights_dir, sync_cfg, vllm_cluster, local_model_dir)
        except Exception as exc:  # noqa: BLE001
            with errors_lock:
                errors[rank] = exc
            logger.exception(
                "[weight_sync] worker %d (%s) sync failed: %s", rank, hostname, exc
            )

    threads = [threading.Thread(target=_task, args=(w,), daemon=True) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        msgs = "; ".join(f"rank {r}: {e}" for r, e in sorted(errors.items()))
        raise RuntimeError(f"Weight sync failed for worker(s): {msgs}")

    logger.info("[weight_sync] All workers reloaded and healthy.")
