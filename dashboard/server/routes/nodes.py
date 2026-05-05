"""Node selection, probing, and inventory routes."""
import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException

from .. import ctx as _ctx
from ..helpers import canonicalize_node_hostname, _self_node, _ssh_aliases_snapshot
from ..models import SelectRequest
from ..node_meta import refresh_node_metadata
from ..redis import redis_mark
from ..ssh_config import _lookup_ssh_entry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/nodes")
async def get_nodes():
    return {
        "discovered":  {_ctx.server_hostname: _self_node(), **dict(_ctx.static_nodes)},
        "selected":    _ctx.node_manager.snapshot_selected(),
        "running":     _ctx.node_manager.snapshot_processes(),
        "usernames":   dict(_ctx.probed),
        "ssh_aliases": _ssh_aliases_snapshot(),
        "node_os":     dict(_ctx.node_os),
    }


@router.get("/api/nodes/{hostname}/probe")
async def probe_node(hostname: str, ssh_user: str = ""):
    from dashboard.node_manager import NodeManager
    canonical = canonicalize_node_hostname(hostname)
    node_ip   = _ctx.static_nodes.get(canonical, {}).get("ip", "")
    ssh_entry = _lookup_ssh_entry(canonical, node_ip)
    target    = ssh_user or ssh_entry.get("alias") or _ctx.ssh_aliases.get(canonical) or canonical
    info      = await NodeManager.probe_metadata(canonical, target)
    if info is None:
        raise HTTPException(502, "Unreachable or SSH key not set up")
    if info.get("username"):
        _ctx.probed[canonical] = info["username"]
        await _ctx.redis.hset("smolcluster:probed", canonical, info["username"])
    _ctx.node_os[canonical] = {
        "os":         info.get("os", ""),
        "os_version": info.get("os_version", ""),
        "machine":    info.get("machine", ""),
    }
    await _ctx.redis.hset("smolcluster:node_os", canonical, json.dumps(_ctx.node_os[canonical]))
    return info


@router.post("/api/nodes/{hostname}/select")
async def select_node(hostname: str, req: SelectRequest):
    canonical = canonicalize_node_hostname(hostname)
    rank      = await _ctx.node_manager.select(canonical, req.ssh_user, req.rank)
    if _ctx.redis:
        await _ctx.redis.hset(
            "smolcluster:selected", canonical,
            json.dumps({"rank": rank, "ssh_user": req.ssh_user}),
        )
        redis_mark(f"selected write: {canonical} rank={rank}", op_key="selected_write")
        logger.info(f"[redis] HSET smolcluster:selected[{canonical}] rank={rank}")
    asyncio.create_task(refresh_node_metadata(canonical, req.ssh_user))
    return {"status": "selected", "rank": rank}


@router.post("/api/nodes/{hostname}/deselect")
async def deselect_node(hostname: str):
    canonical = canonicalize_node_hostname(hostname)
    await _ctx.node_manager.deselect(canonical)
    if _ctx.redis:
        await _ctx.redis.hdel("smolcluster:selected", canonical)
        redis_mark(f"selected delete: {canonical}", op_key="selected_delete")
        logger.info(f"[redis] HDEL smolcluster:selected[{canonical}]")
    return {"status": "deselected"}
