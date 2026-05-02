"""Background broadcasters: events (5 Hz) and log stream (3 Hz)."""
import asyncio
import json
import logging
import time
from pathlib import Path

from . import _ctx
from ._helpers import _self_node, _ssh_aliases_snapshot, canonicalize_log_hostname, _read_json
from ._paths import (
    CLUSTER_LOG_DIR, GRAD_INTERVAL, GRAD_PING, LAST_TOKEN, METRICS_FILE,
    INFERENCE_FILE, TOKEN_INTERVAL, TOKEN_PING,
)
from ._redis import REDIS_EVENTS_KEY, redis_mark, redis_snapshot

logger = logging.getLogger(__name__)

# Log markers emitted by training algorithms to signal a gradient/weight exchange.
_GRAD_SIGNAL_MARKERS = ("[SMOL_METRICS]", "[SMOL_PING]")


async def events_broadcaster() -> None:
    """Compute the full events payload once per 200 ms and cache in Redis.
    All SSE /api/events connections read from the cache key — one compute, N readers."""
    while True:
        try:
            running = _ctx.node_manager.snapshot_processes()
            training_active = any(
                p.get("role") in ("server", "worker", "training_launcher")
                for p in running.values()
            )
            payload = json.dumps({
                "nodes": {
                    "discovered":  {_ctx.server_hostname: _self_node(), **dict(_ctx.static_nodes)},
                    "selected":    _ctx.node_manager.snapshot_selected(),
                    "running":     running,
                    "usernames":   dict(_ctx.probed),
                    "ssh_aliases": _ssh_aliases_snapshot(),
                    "node_os":     dict(_ctx.node_os),
                },
                "training":          _read_json(METRICS_FILE) if training_active else None,
                "connectivity":      _read_json(INFERENCE_FILE),
                "token_ts":          TOKEN_PING.stat().st_mtime   if TOKEN_PING.exists()   else 0,
                "token_text":        LAST_TOKEN.read_text()        if LAST_TOKEN.exists()   else "",
                "token_interval_ms": float(TOKEN_INTERVAL.read_text()) if TOKEN_INTERVAL.exists() else None,
                "grad_ts":           GRAD_PING.stat().st_mtime    if GRAD_PING.exists()    else 0,
                "grad_interval_ms":  float(GRAD_INTERVAL.read_text())  if GRAD_INTERVAL.exists()  else None,
                "redis":             redis_snapshot(),
            })
            await _ctx.redis.set(REDIS_EVENTS_KEY, payload, ex=5)
            redis_mark("events cache write", op_key="events_cache_writes")
        except Exception as exc:
            logger.debug(f"[dashboard] events broadcaster: {exc}")
        await asyncio.sleep(0.2)  # 5 Hz


def _parse_cluster_log_path(path: Path) -> tuple[str, str]:
    stem  = path.stem
    parts = stem.split("__")
    if len(parts) >= 3:
        return parts[0], parts[-2]
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, stem


_LOCAL_LOG_MAX_LINES_PER_TICK = 200


def _read_local_cluster_logs(offsets: dict) -> list[dict]:
    """Tail controller-local tmux logs so local workers stream without Promtail."""
    if not CLUSTER_LOG_DIR.exists():
        offsets.clear()
        return []

    active_paths = {str(p) for p in CLUSTER_LOG_DIR.glob("*__*.log")}
    for stale in [p for p in offsets if p not in active_paths]:
        offsets.pop(stale, None)

    logs: list[dict] = []
    for path in sorted(CLUSTER_LOG_DIR.glob("*__*.log"),
                       key=lambda p: (p.stat().st_mtime_ns, p.name)):
        key = str(path)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if key not in offsets:
            offsets[key] = size  # new connection — tail from EOF, no replay
            continue
        offset = offsets[key]
        if offset > size:
            offset = 0  # file was rotated
        session, hostname = _parse_cluster_log_path(path)
        hostname = canonicalize_log_hostname(hostname, session)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                for raw_line in fh:
                    logs.append({
                        "hostname": hostname, "line": raw_line.rstrip("\n"),
                        "session": session, "ts": time.time(),
                    })
                    if len(logs) >= _LOCAL_LOG_MAX_LINES_PER_TICK:
                        offsets[key] = fh.tell()
                        break
                else:
                    offsets[key] = fh.tell()
        except OSError:
            continue
    return logs


async def log_broadcaster() -> None:
    """Drain NodeManager queue + cluster log files into a Redis Stream."""
    local_log_offsets: dict = {}
    while True:
        try:
            merged: list[dict] = []
            while True:
                try:
                    merged.append(_ctx.node_manager._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            merged.extend(_read_local_cluster_logs(local_log_offsets))
            if merged:
                pipe = _ctx.redis.pipeline()
                for entry in merged:
                    if any(m in entry.get("line", "") for m in _GRAD_SIGNAL_MARKERS):
                        try:
                            GRAD_PING.touch()
                        except Exception:
                            pass
                    pipe.xadd(
                        "smolcluster:logs",
                        {
                            "hostname": entry.get("hostname", ""),
                            "line":     entry.get("line", ""),
                            "session":  entry.get("session", ""),
                            "ts":       str(entry.get("ts") or ""),
                        },
                        maxlen=2000, approximate=True,
                    )
                await pipe.execute()
                redis_mark("logs stream write", op_key="logs_stream_writes", count=len(merged))
        except Exception as exc:
            logger.debug(f"[dashboard] log broadcaster: {exc}")
        await asyncio.sleep(0.35)
