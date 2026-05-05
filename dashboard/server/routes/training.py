"""Training lifecycle routes: start, stop, launch."""
import asyncio
import json
import logging
import subprocess

from fastapi import APIRouter, HTTPException

from .. import ctx as _ctx
from ..helpers import _read_json, build_nodes_info
from ..models import InferenceLaunchRequest, StartRequest
from ..paths import (
    GRAD_INTERVAL, GRAD_PING, GRPO_TRAIN_SCRIPT_FILE,
    INFER_SCRIPT_FILE, METRICS_FILE, TRAIN_CONFIGS_DIR, TRAIN_SCRIPTS_DIR,
)
from ..redis import REDIS_UI_KEY, redis_mark
from dashboard.node_manager import _build_ssh_target

logger = logging.getLogger(__name__)

router = APIRouter()


async def _kill_vllm_on_all_nodes(selected: dict, log_label: str) -> None:
    _kill = (
        "pkill -9 -f 'vllm serve' 2>/dev/null || true; "
        "for port in 8000 8001 8002 8003 8004 8005; do "
        "  lsof -ti :$port 2>/dev/null | xargs kill -9 >/dev/null 2>&1 || true; "
        "done; echo '[vllm] cleanup complete'"
    )
    for hostname, info in selected.items():
        ssh_user = info.get("ssh_user", "")
        target   = _build_ssh_target(ssh_user, hostname) if ssh_user else hostname
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["ssh", "-o", "StrictHostKeyChecking=no",
                 "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                 target, _kill],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if line.strip():
                    _ctx.node_manager._log(log_label, f"[vllm-kill] {target}: {line}")
        except Exception as e:
            _ctx.node_manager._log(log_label, f"[vllm-kill] {target} failed: {e}")


@router.get("/api/training")
async def get_training():
    return _read_json(METRICS_FILE)


@router.post("/api/training/start")
async def start_training(req: StartRequest):
    if not _ctx.node_manager.selected:
        raise HTTPException(400, "No nodes selected")
    try:
        await _ctx.node_manager.start_training(req.algorithm, _ctx.server_hostname)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": "started"}


@router.post("/api/training/stop")
async def stop_training_endpoint():
    running_before = dict(_ctx.node_manager.processes)
    log_label      = next(iter(_ctx.node_manager.processes), _ctx.server_hostname)
    selected       = dict(_ctx.node_manager.selected)

    await _ctx.node_manager.stop_training()
    METRICS_FILE.unlink(missing_ok=True)
    GRAD_PING.unlink(missing_ok=True)
    GRAD_INTERVAL.unlink(missing_ok=True)

    if any(info.get("algorithm") == "grpo" for info in running_before.values()):
        await _ctx.node_manager.run_cleanup_script(str(GRPO_TRAIN_SCRIPT_FILE), log_label)
        await _kill_vllm_on_all_nodes(selected, log_label)

    await _ctx.node_manager.run_cleanup_script(str(INFER_SCRIPT_FILE), log_label)

    if _ctx.redis:
        try:
            raw = await _ctx.redis.get(REDIS_UI_KEY)
            cur = json.loads(raw) if raw else {}
            if isinstance(cur, dict):
                cur["logs"] = []
                await _ctx.redis.set(REDIS_UI_KEY, json.dumps(cur))
                redis_mark("ui-state clear logs on stop", op_key="ui_set")
        except Exception:
            pass
    return {"status": "stopped"}


@router.post("/api/training/launch")
async def launch_training_script(req: InferenceLaunchRequest):
    """Write the algorithm's cluster config YAML and run the training launch script."""
    if not _ctx.node_manager.selected:
        raise HTTPException(400, "No nodes selected")

    snap       = dict(_ctx.static_nodes)
    nodes_info = build_nodes_info(snap)

    if not nodes_info:
        raise HTTPException(400, "No remote nodes selected")

    server_hostname = (
        req.server_hostname
        if req.server_hostname and req.server_hostname in nodes_info
        else min(nodes_info, key=lambda h: nodes_info[h]["rank"])
    )
    try:
        await _ctx.node_manager.launch_training_script(
            algorithm       = req.algorithm,
            server_hostname = server_hostname,
            nodes_info      = nodes_info,
            configs_dir     = TRAIN_CONFIGS_DIR,
            scripts_dir     = TRAIN_SCRIPTS_DIR,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {"status": "launched", "algorithm": req.algorithm, "server": server_hostname}
