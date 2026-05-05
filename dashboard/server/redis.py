"""Redis state, diagnostics, and startup helpers."""
import json
import logging
import os
import subprocess
import time

from . import ctx
from .ssh_config import _lookup_ssh_entry

logger = logging.getLogger(__name__)

REDIS_URL        = os.environ.get("SMOLCLUSTER_REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_EVENTS_KEY = "smolcluster:events"
REDIS_UI_KEY     = "smolcluster:ui_state"

_redis_diag: dict = {
    "enabled": False,
    "status":  "disconnected",
    "url":     REDIS_URL,
    "ops": {
        "selected_restore":    0,
        "selected_write":      0,
        "selected_delete":     0,
        "ui_get":              0,
        "ui_set":              0,
        "events_cache_writes": 0,
        "logs_stream_writes":  0,
    },
    "last_action": "",
    "last_ts":     0.0,
}


def redis_mark(action: str, *, op_key: str = "", count: int = 1) -> None:
    if op_key and op_key in _redis_diag["ops"]:
        _redis_diag["ops"][op_key] += max(0, int(count))
    _redis_diag["last_action"] = action
    _redis_diag["last_ts"]     = time.time()


def redis_snapshot() -> dict:
    return {
        "enabled":     bool(_redis_diag.get("enabled")),
        "status":      _redis_diag.get("status", "unknown"),
        "url":         _redis_diag.get("url", REDIS_URL),
        "ops":         dict(_redis_diag.get("ops", {})),
        "last_action": _redis_diag.get("last_action", ""),
        "last_ts":     _redis_diag.get("last_ts", 0.0),
    }


def ensure_redis_running() -> str:
    """Start Redis via redis-server if not already reachable."""
    try:
        ping = subprocess.run(
            ["redis-cli", "-u", REDIS_URL.replace("/0", ""), "ping"],
            capture_output=True, timeout=1, text=True,
        )
        if ping.returncode == 0 and "PONG" in (ping.stdout or ""):
            return "already-running"
    except Exception:
        pass
    subprocess.run(
        ["redis-server", "--daemonize", "yes",
         "--logfile", "/tmp/redis.log", "--bind", "127.0.0.1"],
        capture_output=True, timeout=10,
    )
    time.sleep(1)
    return "started"


async def restore_state_from_redis() -> None:
    """Restore selected nodes, probed usernames, and node OS info from Redis."""
    from .helpers import canonicalize_node_hostname  # local import avoids cycles at module load

    try:
        selected = await ctx.redis.hgetall("smolcluster:selected")
        restored = skipped = 0
        for hostname, val in selected.items():
            data      = json.loads(val)
            canonical = canonicalize_node_hostname(hostname)
            if canonical != ctx.server_hostname and canonical not in ctx.static_nodes:
                logger.info(f"[dashboard] Redis skipped stale node: {hostname}")
                skipped += 1
                continue
            ctx.node_manager.selected[canonical] = data
            restored += 1
            logger.info(f"[dashboard] Redis restored: {hostname} → {canonical} rank={data.get('rank')}")
        redis_mark(f"restore: restored={restored} skipped={skipped}",
                   op_key="selected_restore", count=restored)
        logger.info(f"[redis] restored selected: {restored} restored, {skipped} skipped")
    except Exception as exc:
        _redis_diag["status"] = "error"
        redis_mark(f"restore failed: {exc}")
        logger.warning(f"[dashboard] Redis restore skipped: {exc}")

    try:
        for hostname, username in (await ctx.redis.hgetall("smolcluster:probed")).items():
            canonical = canonicalize_node_hostname(hostname)
            if canonical and username:
                ctx.probed[canonical] = username
    except Exception:
        pass

    try:
        for hostname, json_str in (await ctx.redis.hgetall("smolcluster:node_os")).items():
            canonical = canonicalize_node_hostname(hostname)
            if canonical:
                try:
                    ctx.node_os[canonical] = json.loads(json_str)
                except Exception:
                    pass
    except Exception:
        pass
