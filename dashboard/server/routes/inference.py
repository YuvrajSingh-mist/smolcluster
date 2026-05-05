"""Inference lifecycle routes: start, stop, launch."""
import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException

from .. import ctx as _ctx
from ..helpers import build_nodes_info
from ..models import InferenceLaunchRequest
from ..paths import INFER_CONFIG_FILE, INFER_SCRIPT_FILE, INFERENCE_FILE, TOKEN_INTERVAL, TOKEN_PING
from ..redis import REDIS_UI_KEY, redis_mark

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/inference/start")
async def start_inference():
    if not _ctx.node_manager.selected:
        raise HTTPException(400, "No nodes selected")
    try:
        await _ctx.node_manager.start_inference(_ctx.server_hostname)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": "started"}


@router.post("/api/inference/stop")
async def stop_inference():
    log_label = next(iter(_ctx.node_manager.processes), _ctx.server_hostname)
    await _ctx.node_manager.stop_training()
    INFERENCE_FILE.unlink(missing_ok=True)
    TOKEN_PING.unlink(missing_ok=True)
    TOKEN_INTERVAL.unlink(missing_ok=True)
    asyncio.create_task(
        _ctx.node_manager.run_cleanup_script(str(INFER_SCRIPT_FILE), log_label)
    )
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


@router.post("/api/inference/launch")
async def launch_inference_script(req: InferenceLaunchRequest):
    """Write cluster_config_inference.yaml and run launch_inference.sh."""
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
        await _ctx.node_manager.launch_inference_script(
            algorithm       = req.algorithm,
            server_hostname = server_hostname,
            nodes_info      = nodes_info,
            config_path     = str(INFER_CONFIG_FILE),
            script_path     = str(INFER_SCRIPT_FILE),
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {"status": "launched", "algorithm": req.algorithm, "server": server_hostname}
