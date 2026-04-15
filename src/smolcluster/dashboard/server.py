"""
smolcluster Dashboard — FastAPI backend.

Endpoints:
    GET  /                              → index.html
    GET  /api/nodes                     → discovered + selected + running
    GET  /api/events                    → SSE: nodes + training + inference (1 Hz)
    GET  /api/logs                      → SSE: live log lines (3 Hz)
    GET  /api/nodes/{host}/probe        → SSH whoami
    POST /api/nodes/{host}/select       → mark node ready
    POST /api/nodes/{host}/deselect     → unmark
    POST /api/training/start            → launch server + workers
    POST /api/training/stop             → kill everything
    POST /api/inference/start           → launch infer server + workers
    POST /api/inference/stop            → kill + clear file
    POST /api/connectivity/check        → TCP port-22 check (no SSH keys needed)
    GET  /api/training                  → latest metrics JSON
    POST /chat                          → proxy to inference API chat (SSE)
"""

import asyncio
import json
import logging
import os
import platform
import re
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from smolcluster.dashboard.node_manager import NodeManager

logger = logging.getLogger(__name__)

FRONTEND_DIR   = Path(__file__).parent / "frontend"
METRICS_FILE   = Path("/tmp/smolcluster_metrics.json")
INFERENCE_FILE = Path("/tmp/smolcluster_inference.json")
TOKEN_PING     = Path("/tmp/smolcluster_token_ping")
LAST_TOKEN     = Path("/tmp/smolcluster_last_token")
TOKEN_INTERVAL = Path("/tmp/smolcluster_token_interval_ms")  # real inter-token ms written by api.py
GRAD_PING      = Path("/tmp/smolcluster_grad_ping")
GRAD_INTERVAL  = Path("/tmp/smolcluster_grad_interval_ms")   # real inter-step ms written by training servers
LOKI_BASE_URL  = os.environ.get("SMOLCLUSTER_LOKI_URL", "http://127.0.0.1:3100")
CLUSTER_LOG_DIR = Path(__file__).resolve().parents[3] / "logging" / "cluster-logs"


# ── SSH config parsing ─────────────────────────────────────────────────────────
def parse_ssh_config() -> dict:
    """
    Parse ~/.ssh/config and return
    {key: {alias, user, hostname}} where key may be alias, HostName, or HostName variants.
    Skips wildcard/glob Host entries. Called once at import time.
    """
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        return {}
    result: dict = {}
    current_host: Optional[str] = None
    current: dict = {}

    def _flush():
        if current_host and "*" not in current_host and "?" not in current_host:
            alias = current_host.split()[0]
            host_name = current.get("Hostname", "")
            entry = {
                "alias": alias,
                "user": current.get("User", ""),
                "hostname": host_name,
            }

            keys = {alias}
            if host_name:
                keys.add(host_name)
                if host_name.endswith(".local"):
                    keys.add(host_name.removesuffix(".local"))
                else:
                    keys.add(f"{host_name}.local")

            for key in keys:
                result[key] = entry

    for raw in config_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        k, v = parts[0].lower(), parts[1].strip()
        if k == "host":
            _flush()
            current_host, current = v, {}
        elif k == "hostname":
            current["Hostname"] = v
        elif k == "user":
            current["User"] = v
    _flush()
    return result


_SSH_CONFIG:  dict = parse_ssh_config()   # ip/hostname → {alias, user}
_ssh_aliases: dict = {}                   # hostname → SSH alias (usually the alias itself)
_static_nodes: dict = {}                  # hostname → node info (ssh-config inventory)


def _lookup_ssh_entry(hostname: str, node_ip: str) -> dict:
    """
    Resolve a discovered node to an SSH config entry.

    Priority:
    1) discovered IP
    2) hostname.local
    3) bare hostname
    4) heuristic jetson alias mapping (jetson-nano1 -> jetson, jetson-nano2 -> jetson2)
    """
    for key in (node_ip, f"{hostname}.local", hostname):
        if key and key in _SSH_CONFIG:
            return _SSH_CONFIG[key]

    # Heuristic for common host naming on Jetson clusters.
    m = re.search(r"(\d+)$", hostname)
    if m:
        idx = int(m.group(1))
        candidates = [f"jetson{idx}"]
        if idx == 1:
            candidates.insert(0, "jetson")
        for alias in candidates:
            if alias in _SSH_CONFIG:
                return _SSH_CONFIG[alias]

    if "jetson" in hostname.lower() and "jetson" in _SSH_CONFIG:
        return _SSH_CONFIG["jetson"]

    return {}


def _get_local_ip() -> str:
    """Best-effort: get the default-route IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return ""


def _get_server_alias(server_hostname: str) -> str:
    """Return the SSH config alias for the local server, or the hostname itself."""
    candidates = []

    # Include all local interface IPs (important when hostname resolves to LAN
    # while SSH config points at another local interface such as 10.x.x.x).
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        candidates.extend([ip for ip in out.split() if ip])
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except Exception:
        pass

    local_ip = _get_local_ip()
    if local_ip:
        candidates.append(local_ip)

    candidates.extend([server_hostname, f"{server_hostname}.local"])

    for key in candidates:
        if key in _SSH_CONFIG:
            return _SSH_CONFIG[key]["alias"]
    return server_hostname


def _build_static_nodes_inventory(server_hostname: str, local_ips: set) -> dict:
    """Build SSH-config-backed node inventory keyed by SSH alias."""
    by_alias: dict = {}
    for entry in _SSH_CONFIG.values():
        alias = (entry.get("alias") or "").strip()
        if not alias or alias in by_alias:
            continue
        by_alias[alias] = entry

    nodes: dict = {}
    for alias, entry in by_alias.items():
        host_name = (entry.get("hostname") or "").strip()
        host_name_no_local = host_name.removesuffix(".local")
        if alias == server_hostname or host_name in local_ips or host_name_no_local == server_hostname:
            continue
        nodes[alias] = {
            "hostname": alias,
            "alias": alias,
            "ip": host_name,
            "port": 22,
            "os": "",
            "os_version": "",
            "machine": "",
            "role": "available",
            "source": "ssh_config",
        }
    return nodes


def _local_node_metadata() -> dict:
    return {
        "os": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "machine": platform.machine(),
    }


def _canonicalize_node_hostname(hostname: str) -> str:
    name = (hostname or "").strip().removesuffix(".local")
    if not name:
        return name
    if name == _server_hostname:
        return name
    if name in _static_nodes:
        return name

    server_alias = (_get_server_alias(_server_hostname) or "").removesuffix(".local")
    if name == server_alias:
        return _server_hostname

    ssh_entry = _SSH_CONFIG.get(name)
    alias = (ssh_entry or {}).get("alias", "")
    if alias in _static_nodes:
        return alias

    return name

# ── App state ─────────────────────────────────────────────────────────────────
node_manager:     NodeManager
_server_hostname: str = ""
_redis:           aioredis.Redis = None

REDIS_URL = os.environ.get("SMOLCLUSTER_REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_EVENTS_KEY = "smolcluster:events"

_redis_diag: dict = {
    "enabled": False,
    "status": "disconnected",
    "url": REDIS_URL,
    "ops": {
        "selected_restore": 0,
        "selected_write": 0,
        "selected_delete": 0,
        "ui_get": 0,
        "ui_set": 0,
        "events_cache_writes": 0,
        "logs_stream_writes": 0,
    },
    "last_action": "",
    "last_ts": 0.0,
}


def _redis_mark(action: str, *, op_key: Optional[str] = None, count: int = 1) -> None:
    if op_key and op_key in _redis_diag["ops"]:
        _redis_diag["ops"][op_key] += max(0, int(count))
    _redis_diag["last_action"] = action
    _redis_diag["last_ts"] = time.time()


def _redis_snapshot() -> dict:
    return {
        "enabled": bool(_redis_diag.get("enabled", False)),
        "status": _redis_diag.get("status", "unknown"),
        "url": _redis_diag.get("url", REDIS_URL),
        "ops": dict(_redis_diag.get("ops", {})),
        "last_action": _redis_diag.get("last_action", ""),
        "last_ts": _redis_diag.get("last_ts", 0.0),
    }


def _ensure_redis_running() -> str:
    """Start Redis via redis-server if not already reachable."""
    try:
        ping = subprocess.run(
            ["redis-cli", "-u", REDIS_URL.replace("/0", ""), "ping"],
            capture_output=True,
            timeout=1,
            text=True,
        )
        if ping.returncode == 0 and "PONG" in (ping.stdout or ""):
            return "already-running"
    except Exception:
        pass
    
    subprocess.run(["redis-server", "--daemonize", "yes",
             "--logfile", "/tmp/redis.log", "--bind", "127.0.0.1"],
            capture_output=True, timeout=10)
    
    time.sleep(1)
    return "started"


async def _restore_state_from_redis():
    """Restore selected nodes from Redis so the dashboard survives restarts."""
    try:
        selected = await _redis.hgetall("smolcluster:selected")
        restored = 0
        skipped = 0
        for hostname, val in selected.items():
            data = json.loads(val)
            canonical = _canonicalize_node_hostname(hostname)
            if canonical != _server_hostname and canonical not in _static_nodes:
                logger.info(f"[dashboard] Redis skipped stale node: {hostname}")
                skipped += 1
                continue
            node_manager.selected[canonical] = data
            restored += 1
            logger.info(f"[dashboard] Redis restored: {hostname} -> {canonical} rank={data.get('rank')}")
        _redis_mark(f"restore selected: restored={restored} skipped={skipped}", op_key="selected_restore", count=restored)
        logger.info(f"[redis] restored selected nodes: restored={restored} skipped={skipped}")
    except Exception as exc:
        _redis_diag["status"] = "error"
        _redis_mark(f"restore selected failed: {exc}")
        logger.warning(f"[dashboard] Redis restore skipped: {exc}")


async def _refresh_node_metadata(hostname: str, ssh_hint: str = ""):
    canonical = _canonicalize_node_hostname(hostname)
    if not canonical:
        return

    if canonical == _server_hostname:
        _node_os[canonical] = _local_node_metadata()
        return

    node_ip = _static_nodes.get(canonical, {}).get("ip", "")
    ssh_entry = _lookup_ssh_entry(canonical, node_ip)
    target_hint = ssh_hint or ssh_entry.get("alias") or _ssh_aliases.get(canonical) or canonical
    info = await NodeManager.probe_metadata(canonical, target_hint)
    if not info:
        return

    username = (info.get("username") or "").strip()
    if username:
        _probed[canonical] = username

    os_info = {
        "os": (info.get("os") or "").strip(),
        "os_version": (info.get("os_version") or "").strip(),
        "machine": (info.get("machine") or "").strip(),
    }
    if any(os_info.values()):
        _node_os[canonical] = os_info


async def _prime_node_metadata():
    await _refresh_node_metadata(_server_hostname)
    tasks = [asyncio.create_task(_refresh_node_metadata(hostname)) for hostname in _static_nodes]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global node_manager, _server_hostname, _redis, _static_nodes
    redis_boot = _ensure_redis_running()
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _redis_diag["enabled"] = True
    _redis_diag["status"] = "connected"
    _redis_mark(f"connected ({redis_boot})")
    logger.info(f"[redis] connected ({redis_boot}) url={REDIS_URL}")
    node_manager = NodeManager()
    _server_hostname = socket.gethostname().removesuffix(".local")
    _static_nodes = _build_static_nodes_inventory(_server_hostname, _LOCAL_IPS)
    _ssh_aliases.clear()
    _ssh_aliases.update({h: h for h in _static_nodes})
    _node_os[_server_hostname] = _local_node_metadata()
    for h in _static_nodes:
        _probed[h] = _probed.get(h) or _lookup_ssh_entry(h, _static_nodes[h].get("ip", "")).get("user", "")
    await _restore_state_from_redis()
    logger.info(f"[dashboard] http://{_server_hostname}.local:9090")
    broadcast_task = asyncio.create_task(_events_broadcaster())
    log_task       = asyncio.create_task(_log_broadcaster())
    metadata_task  = asyncio.create_task(_prime_node_metadata())
    yield
    broadcast_task.cancel()
    log_task.cancel()
    metadata_task.cancel()
    await asyncio.gather(broadcast_task, log_task, metadata_task, return_exceptions=True)
    await node_manager.stop_all()
    await _redis.aclose()


app = FastAPI(title="smolcluster Dashboard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Node metadata cache ─────────────────────────────────────────────────────────
_probed:   dict = {}  # hostname → SSH username
_node_os:  dict = {}  # hostname → {os, os_version, machine}

# Cache local IPs at import time — used by _is_local_node to avoid re-probing.
def _collect_local_ips() -> set:
    ips: set = set()
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        for ip in out.split():
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.add(ip)
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return ips

_LOCAL_IPS: set = _collect_local_ips()


# ── Request models ─────────────────────────────────────────────────────────────
class SelectRequest(BaseModel):
    ssh_user: str = ""
    rank: Optional[int] = None

class StartRequest(BaseModel):
    algorithm: str = "syncps"

class InferenceLaunchRequest(BaseModel):
    algorithm: str = "syncps"
    server_hostname: str = ""   # which selected node is the server/rank-0


def _self_node() -> dict:
    """Build the node entry for the local (server) machine."""
    alias = _get_server_alias(_server_hostname)
    return {
        "hostname":   _server_hostname,
        "alias":      alias,
        "ip":         _get_local_ip() or "127.0.0.1",
        "port":       9090,
        "os":         _node_os.get(_server_hostname, {}).get("os", platform.system()),
        "os_version": _node_os.get(_server_hostname, {}).get("os_version",
                          platform.mac_ver()[0] or platform.release()),
        "machine":    _node_os.get(_server_hostname, {}).get("machine", platform.machine()),
        "role":       "server",
        "source":     "local",
    }


def _ssh_aliases_snapshot() -> dict:
    """Return alias map with local server alias included."""
    aliases = dict(_ssh_aliases)
    aliases[_server_hostname] = _get_server_alias(_server_hostname)
    return aliases


def _looks_like_server_session(session: str) -> bool:
    session = (session or "").strip().lower()
    return bool(re.search(r"(^|[-_])server($|[-_])", session))


def _canonicalize_log_hostname(raw_hostname: str, session: str = "") -> str:
    """Resolve log host labels back to the dashboard's canonical node hostname."""
    hostname = (raw_hostname or "").strip().removesuffix(".local")
    if not hostname:
        return _server_hostname if _looks_like_server_session(session) else "unknown"

    if hostname == _server_hostname:
        return hostname

    server_alias = (_get_server_alias(_server_hostname) or "").removesuffix(".local")
    if _looks_like_server_session(session) and hostname == server_alias:
        return _server_hostname

    known_hosts = {
        _server_hostname,
        *_static_nodes.keys(),
        *node_manager.snapshot_selected().keys(),
        *node_manager.snapshot_processes().keys(),
    }
    if hostname in known_hosts:
        return hostname

    alias_matches = [
        canonical
        for canonical, alias in _ssh_aliases_snapshot().items()
        if (alias or "").strip().removesuffix(".local") == hostname
    ]
    if len(alias_matches) == 1:
        return alias_matches[0]
    if len(alias_matches) > 1 and _looks_like_server_session(session) and _server_hostname in alias_matches:
        return _server_hostname

    return hostname


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/nodes")
async def get_nodes():
    discovered = {_server_hostname: _self_node(), **dict(_static_nodes)}
    return {
        "discovered":  discovered,
        "selected":    node_manager.snapshot_selected(),
        "running":     node_manager.snapshot_processes(),
        "usernames":   dict(_probed),
        "ssh_aliases": _ssh_aliases_snapshot(),
        "node_os":     dict(_node_os),
    }


@app.get("/api/nodes/{hostname}/probe")
async def probe_node(hostname: str, ssh_user: str = ""):
    canonical = _canonicalize_node_hostname(hostname)
    node_ip = _static_nodes.get(canonical, {}).get("ip", "")
    ssh_entry = _lookup_ssh_entry(canonical, node_ip)
    target_hint = ssh_user or ssh_entry.get("alias") or _ssh_aliases.get(canonical) or canonical
    info = await NodeManager.probe_metadata(canonical, target_hint)
    if info is None:
        raise HTTPException(502, "Unreachable or SSH key not set up")
    if info.get("username"):
        _probed[canonical] = info["username"]
    _node_os[canonical] = {
        "os": info.get("os", ""),
        "os_version": info.get("os_version", ""),
        "machine": info.get("machine", ""),
    }
    return info


@app.post("/api/nodes/{hostname}/select")
async def select_node(hostname: str, req: SelectRequest):
    canonical = _canonicalize_node_hostname(hostname)
    rank = await node_manager.select(canonical, req.ssh_user, req.rank)
    if _redis:
        await _redis.hset("smolcluster:selected", canonical,
                          json.dumps({"rank": rank, "ssh_user": req.ssh_user}))
        _redis_mark(f"selected write: {canonical} rank={rank}", op_key="selected_write")
        logger.info(f"[redis] HSET smolcluster:selected[{canonical}] rank={rank}")
    asyncio.create_task(_refresh_node_metadata(canonical, req.ssh_user))
    return {"status": "selected", "rank": rank}


@app.post("/api/nodes/{hostname}/deselect")
async def deselect_node(hostname: str):
    canonical = _canonicalize_node_hostname(hostname)
    await node_manager.deselect(canonical)
    if _redis:
        await _redis.hdel("smolcluster:selected", canonical)
        _redis_mark(f"selected delete: {canonical}", op_key="selected_delete")
        logger.info(f"[redis] HDEL smolcluster:selected[{canonical}]")
    return {"status": "deselected"}


@app.post("/api/training/start")
async def start_training(req: StartRequest):
    if not node_manager.selected:
        raise HTTPException(400, "No nodes selected")
    try:
        await node_manager.start_training(req.algorithm, _server_hostname)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": "started"}


@app.post("/api/training/stop")
async def stop_training_endpoint():
    running_before_stop = dict(node_manager.processes)
    log_label = next(iter(node_manager.processes), _server_hostname)
    await node_manager.stop_training()
    # Clear stale metric and ping files so a fresh page load or new run starts clean.
    METRICS_FILE.unlink(missing_ok=True)
    GRAD_PING.unlink(missing_ok=True)
    GRAD_INTERVAL.unlink(missing_ok=True)
    if any(info.get("algorithm") == "grpo" for info in running_before_stop.values()):
        asyncio.create_task(
            node_manager.run_cleanup_script(str(GRPO_TRAIN_SCRIPT_FILE), log_label)
        )
    # Also run inference cleanup in case sessions from a previous inference run survived
    asyncio.create_task(
        node_manager.run_cleanup_script(str(INFER_SCRIPT_FILE), log_label)
    )
    if _redis:
        try:
            raw = await _redis.get(REDIS_UI_KEY)
            cur = json.loads(raw) if raw else {}
            if isinstance(cur, dict):
                cur["logs"] = []
                await _redis.set(REDIS_UI_KEY, json.dumps(cur))
                _redis_mark("ui-state clear logs on stop", op_key="ui_set")
                logger.info(f"[redis] SET {REDIS_UI_KEY} logs cleared on training stop")
        except Exception:
            pass
    return {"status": "stopped"}


@app.post("/api/inference/start")
async def start_inference():
    if not node_manager.selected:
        raise HTTPException(400, "No nodes selected")
    try:
        await node_manager.start_inference(_server_hostname)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": "started"}


@app.post("/api/inference/stop")
async def stop_inference():
    log_label = next(iter(node_manager.processes), _server_hostname)
    await node_manager.stop_training()
    INFERENCE_FILE.unlink(missing_ok=True)
    TOKEN_PING.unlink(missing_ok=True)
    TOKEN_INTERVAL.unlink(missing_ok=True)
    # Run launch_inference.sh --cleanup and stream its output to the log buffer
    asyncio.create_task(
        node_manager.run_cleanup_script(str(INFER_SCRIPT_FILE), log_label)
    )
    if _redis:
        try:
            raw = await _redis.get(REDIS_UI_KEY)
            cur = json.loads(raw) if raw else {}
            if isinstance(cur, dict):
                cur["logs"] = []
                await _redis.set(REDIS_UI_KEY, json.dumps(cur))
                _redis_mark("ui-state clear logs on stop", op_key="ui_set")
                logger.info(f"[redis] SET {REDIS_UI_KEY} logs cleared on inference stop")
        except Exception:
            pass
    return {"status": "stopped"}


INFER_CONFIG_FILE = (Path(__file__).parent.parent /
                     "configs" / "inference" / "cluster_config_inference.yaml")
INFER_SCRIPT_FILE = (Path(__file__).parent.parent.parent.parent /
                     "scripts" / "inference" / "launch_inference.sh")
GRPO_TRAIN_SCRIPT_FILE = (Path(__file__).parent.parent / "applications" /
                          "reasoning" / "grpo" / "scripts" / "launch_grpo_train.sh")

TRAIN_CONFIGS_DIR = str(Path(__file__).parent.parent / "configs")
TRAIN_SCRIPTS_DIR = str(Path(__file__).parent.parent.parent.parent / "scripts" / "training")


@app.post("/api/inference/launch")
async def launch_inference_script(req: InferenceLaunchRequest):
    """Write cluster_config_inference.yaml and run launch_inference.sh."""
    if not node_manager.selected:
        raise HTTPException(400, "No nodes selected")

    algorithm = req.algorithm
    snap      = dict(_static_nodes)

    # Build nodes_info from selected nodes only — local machine is never included
    # (scripts run locally and rsync TO remote nodes; including self breaks rsync)
    nodes_info: dict = {}
    for hostname, sel in node_manager.selected.items():
        if hostname == _server_hostname:
            node_ip = _get_local_ip() or _self_node().get("ip", "")
            ssh_entry = _lookup_ssh_entry(hostname, node_ip)
            local_alias = _get_server_alias(hostname)
        else:
            node_ip = snap.get(hostname, {}).get("ip", "")
            ssh_entry = _lookup_ssh_entry(hostname, node_ip)
            local_alias = ""
        # Alias = SSH Host entry (e.g. "mini2") — try four ways in order:
        # 1. By LAN IP  (SSH config has HostName 10.x.x.x)
        # 2. By .local FQDN  (SSH config has HostName macmini3-5.local)
        # 3. By bare hostname  (SSH config has HostName macmini3-5)
        # 4. Alias cache populated from SSH inventory
        # Never use ssh_user (username) as the alias — they're different things.
        alias = (
            ssh_entry.get("alias")
            or local_alias
            or _ssh_aliases.get(hostname)
            or hostname
        )
        preferred_ip = ssh_entry.get("hostname") or node_ip
        # Prefer probed username; fall back to ~/.ssh/config User and selected ssh_user.
        user  = (
            _probed.get(hostname)
            or ssh_entry.get("user")
            or sel.get("ssh_user", "")
            or ""
        )
        nodes_info[hostname] = {
            "ssh_alias": alias,
            "user":      user,
            "rank":      sel["rank"],
            "ip":        preferred_ip,
        }

    if not nodes_info:
        raise HTTPException(400, "No remote nodes selected")

    # Determine server: user-picked (req.server_hostname) or lowest-rank node
    server_hostname = (
        req.server_hostname
        if req.server_hostname and req.server_hostname in nodes_info
        else min(nodes_info, key=lambda h: nodes_info[h]["rank"])
    )

    try:
        await node_manager.launch_inference_script(
            algorithm        = algorithm,
            server_hostname  = server_hostname,
            nodes_info       = nodes_info,
            config_path      = str(INFER_CONFIG_FILE),
            script_path      = str(INFER_SCRIPT_FILE),
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {"status": "launched", "algorithm": algorithm, "server": server_hostname}


@app.post("/api/training/launch")
async def launch_training_script(req: InferenceLaunchRequest):
    """Write the algorithm's cluster config YAML and run the training launch script."""
    if not node_manager.selected:
        raise HTTPException(400, "No nodes selected")

    algorithm = req.algorithm
    snap      = dict(_static_nodes)

    nodes_info: dict = {}
    for hostname, sel in node_manager.selected.items():
        if hostname == _server_hostname:
            node_ip = _get_local_ip() or _self_node().get("ip", "")
            ssh_entry = _lookup_ssh_entry(hostname, node_ip)
            local_alias = _get_server_alias(hostname)
        else:
            node_ip = snap.get(hostname, {}).get("ip", "")
            ssh_entry = _lookup_ssh_entry(hostname, node_ip)
            local_alias = ""
        alias = (
            ssh_entry.get("alias")
            or local_alias
            or _ssh_aliases.get(hostname)
            or hostname
        )
        preferred_ip = ssh_entry.get("hostname") or node_ip
        # Prefer probed username; fall back to ~/.ssh/config User and selected ssh_user.
        user = (
            _probed.get(hostname)
            or ssh_entry.get("user")
            or sel.get("ssh_user", "")
            or ""
        )
        nodes_info[hostname] = {
            "ssh_alias": alias,
            "user":      user,
            "rank":      sel["rank"],
            "ip":        preferred_ip,
        }

    if not nodes_info:
        raise HTTPException(400, "No remote nodes selected")

    server_hostname = (
        req.server_hostname
        if req.server_hostname and req.server_hostname in nodes_info
        else min(nodes_info, key=lambda h: nodes_info[h]["rank"])
    )

    try:
        await node_manager.launch_training_script(
            algorithm       = algorithm,
            server_hostname = server_hostname,
            nodes_info      = nodes_info,
            configs_dir     = TRAIN_CONFIGS_DIR,
            scripts_dir     = TRAIN_SCRIPTS_DIR,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {"status": "launched", "algorithm": algorithm, "server": server_hostname}


@app.post("/api/connectivity/check")
async def connectivity_check():
    """TCP port-22 probe — works without SSH keys."""
    selected = node_manager.snapshot_selected()
    if not selected:
        raise HTTPException(400, "No nodes selected")
    asyncio.create_task(_run_tcp_checks(selected, dict(_static_nodes)))
    return {"status": "checking"}


@app.get("/api/training")
async def get_training():
    return _read_json(METRICS_FILE)


REDIS_UI_KEY = "smolcluster:ui_state"


@app.get("/api/ui-state")
async def get_ui_state():
    """Return persisted UI state from Redis."""
    if _redis:
        try:
            raw = await _redis.get(REDIS_UI_KEY)
            _redis_mark("ui-state get", op_key="ui_get")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return {}


@app.post("/api/ui-state")
async def post_ui_state(request: Request):
    """Merge a UI state patch into Redis."""
    if _redis:
        try:
            patch = await request.json()
            raw = await _redis.get(REDIS_UI_KEY)
            cur = json.loads(raw) if raw else {}
            cur.update(patch)
            await _redis.set(REDIS_UI_KEY, json.dumps(cur))
            keys = ",".join(sorted(str(k) for k in patch.keys())) if isinstance(patch, dict) else "unknown"
            _redis_mark(f"ui-state set keys={keys}", op_key="ui_set")
            logger.info(f"[redis] SET {REDIS_UI_KEY} keys={keys}")
        except Exception:
            pass
    return {"ok": True}


# ── Background broadcasters (started in lifespan) ─────────────────────────────
async def _events_broadcaster():
    """Compute the full events payload once per second and cache in Redis.
    All SSE /api/events connections read from the cache key instead of
    recomputing independently — one compute, N readers."""
    while True:
        try:
            _running_procs = node_manager.snapshot_processes()
            _training_active = any(
                p.get("role") in ("server", "worker", "training_launcher")
                for p in _running_procs.values()
            )
            payload = json.dumps({
                "nodes": {
                    "discovered": {_server_hostname: _self_node(), **dict(_static_nodes)},
                    "selected": node_manager.snapshot_selected(),
                    "running": _running_procs,
                    "usernames": dict(_probed),
                    "ssh_aliases": _ssh_aliases_snapshot(),
                    "node_os": dict(_node_os),
                },
                # Don't serve stale METRICS_FILE when no training processes are running.
                # A fresh page load after a stopped run must not show old values.
                "training": _read_json(METRICS_FILE) if _training_active else None,
                "connectivity": _read_json(INFERENCE_FILE),
                "token_ts": TOKEN_PING.stat().st_mtime if TOKEN_PING.exists() else 0,
                "token_text": LAST_TOKEN.read_text() if LAST_TOKEN.exists() else "",
                "token_interval_ms": float(TOKEN_INTERVAL.read_text()) if TOKEN_INTERVAL.exists() else None,
                "grad_ts": GRAD_PING.stat().st_mtime if GRAD_PING.exists() else 0,
                "grad_interval_ms": float(GRAD_INTERVAL.read_text()) if GRAD_INTERVAL.exists() else None,
                "redis": _redis_snapshot(),
            })
            await _redis.set(REDIS_EVENTS_KEY, payload, ex=5)
            _redis_mark("events cache write", op_key="events_cache_writes")
        except Exception as exc:
            logger.debug(f"[dashboard] events broadcaster: {exc}")
        await asyncio.sleep(0.2)  # 5 Hz — fast enough to track token-by-token inference


# Log markers emitted by training algorithms to signal a gradient/weight exchange.
_GRAD_SIGNAL_MARKERS = ("[SMOL_METRICS]", "[SMOL_PING]")


async def _log_broadcaster():
    """Drain NodeManager in-memory logs + cluster log files into a Redis Stream.
    All SSE /api/logs connections consume via XREAD BLOCK — no file reads per
    connection and no history replay on reconnect."""
    last_seq = 0
    local_log_offsets: dict[str, int] = {}
    while True:
        try:
            lines = node_manager.logs_since(last_seq)
            local_lines = _read_local_cluster_logs(local_log_offsets)
            merged: list[dict] = []
            if lines:
                last_seq = lines[-1]["seq"]
                merged.extend(lines)
            merged.extend(local_lines)
            if merged:
                pipe = _redis.pipeline()
                for entry in merged:
                    # Touch GRAD_PING on the dashboard machine whenever any node
                    # logs a gradient/weight exchange marker.  This is the only
                    # reliable path for remote workers (FSDP, EP) whose /tmp/ is
                    # not on the controller machine.
                    if any(m in entry.get("line", "") for m in _GRAD_SIGNAL_MARKERS):
                        try: GRAD_PING.touch()
                        except Exception: pass
                    pipe.xadd(
                        "smolcluster:logs",
                        {
                            "hostname": entry.get("hostname", ""),
                            "line":     entry.get("line", ""),
                            "session":  entry.get("session", ""),
                            "ts":       str(entry.get("ts") or ""),
                        },
                        maxlen=2000,
                        approximate=True,
                    )
                await pipe.execute()
                _redis_mark("logs stream write", op_key="logs_stream_writes", count=len(merged))
        except Exception as exc:
            logger.debug(f"[dashboard] log broadcaster: {exc}")
        await asyncio.sleep(0.35)


# ── SSE: state (1 Hz) ──────────────────────────────────────────────────────────
@app.get("/api/events")
async def sse_events(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                raw = await _redis.get(REDIS_EVENTS_KEY)
                if raw:
                    yield f"data: {raw}\n\n"
            except Exception:
                pass
            await asyncio.sleep(0.15)  # 6-7 Hz for snappy token / packet updates
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── SSE: logs (3 Hz) ───────────────────────────────────────────────────────────
async def _fetch_loki_logs(start_ns: int) -> tuple[list[dict], int]:
    """Fetch centralized worker/server logs from Loki from the given timestamp."""
    query = '{job="smolcluster"}'
    end_ns = time.time_ns()
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(
                f"{LOKI_BASE_URL}/loki/api/v1/query_range",
                params={
                    "query": query,
                    "start": str(start_ns),
                    "end": str(end_ns),
                    "direction": "forward",
                    "limit": "500",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return [], start_ns

    results = payload.get("data", {}).get("result", [])
    logs: list[dict] = []
    max_seen_ns = start_ns
    for stream in results:
        labels = stream.get("stream", {}) or {}
        session = labels.get("session") or ""
        raw_hostname = labels.get("host") or labels.get("hostname") or labels.get("job") or "remote"
        hostname = _canonicalize_log_hostname(raw_hostname, session)
        for ts_text, line in stream.get("values", []) or []:
            try:
                ts_ns = int(ts_text)
            except (TypeError, ValueError):
                continue
            max_seen_ns = max(max_seen_ns, ts_ns)
            logs.append({
                "hostname": hostname,
                "line": line,
                "session": session,
                "ts": ts_ns / 1_000_000_000,
            })

    logs.sort(key=lambda item: (item["ts"], item["hostname"], item["line"]))
    return logs, max_seen_ns + 1


def _parse_cluster_log_path(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        session, hostname = stem.rsplit("__", 1)
        return session, hostname
    return stem, stem


_LOCAL_LOG_MAX_LINES_PER_TICK = 200


def _read_local_cluster_logs(offsets: dict[str, int]) -> list[dict]:
    """Tail controller-local tmux logs so local workers stream even without Promtail."""
    if not CLUSTER_LOG_DIR.exists():
        offsets.clear()
        return []

    active_paths = {str(path) for path in CLUSTER_LOG_DIR.glob("*__*.log")}
    for stale_path in [path for path in offsets if path not in active_paths]:
        offsets.pop(stale_path, None)

    logs: list[dict] = []
    for path in sorted(CLUSTER_LOG_DIR.glob("*__*.log"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        key = str(path)
        try:
            size = path.stat().st_size
        except OSError:
            continue

        if key not in offsets:
            # New SSE connection: start at EOF so we only stream new lines,
            # never replay the full history (can be many MB).
            offsets[key] = size
            continue

        offset = offsets[key]
        if offset > size:
            # File was rotated/truncated — restart from beginning.
            offset = 0

        session, hostname = _parse_cluster_log_path(path)
        hostname = _canonicalize_log_hostname(hostname, session)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for raw_line in handle:
                    logs.append({
                        "hostname": hostname,
                        "line": raw_line.rstrip("\n"),
                        "session": session,
                        "ts": time.time(),
                    })
                    if len(logs) >= _LOCAL_LOG_MAX_LINES_PER_TICK:
                        offsets[key] = handle.tell()
                        break
                else:
                    offsets[key] = handle.tell()
        except OSError:
            continue

    return logs


@app.get("/api/logs")
async def sse_logs(request: Request):
    async def gen():
        last_id = "$"   # only new log entries from this connection onward
        while True:
            if await request.is_disconnected():
                break
            try:
                results = await _redis.xread(
                    {"smolcluster:logs": last_id}, count=200, block=400
                )
                if results:
                    _, entries = results[0]
                    if entries:
                        last_id = entries[-1][0]
                        lines = [
                            {
                                "hostname": e["hostname"],
                                "line":     e["line"],
                                "session":  e.get("session", ""),
                                "ts":       float(e.get("ts") or 0),
                            }
                            for _, e in entries
                        ]
                        yield f"data: {json.dumps(lines)}\n\n"
            except Exception:
                await asyncio.sleep(0.35)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── TCP connectivity check (background task) ───────────────────────────────────
def _resolve_connectivity_target(hostname: str, snap: dict) -> str:
    """
    Resolve the best network target for a node connectivity probe.

    Priority:
    1) SSH config HostName/IP (works for aliases like jetson2)
    2) discovered node IP
    3) hostname.local
    """
    node_ip = snap.get(hostname, {}).get("ip", "")
    ssh_entry = _lookup_ssh_entry(hostname, node_ip)
    if ssh_entry.get("hostname"):
        return ssh_entry["hostname"]
    if node_ip:
        return node_ip
    return f"{hostname}.local"


async def _run_tcp_checks(selected: dict, snap: dict):
    total = len(selected)
    results = []
    INFERENCE_FILE.write_text(json.dumps({
        "mode": "connectivity", "status": "checking",
        "results": [], "total": total,
        "message": f"Checking {total} node(s)…",
    }))
    for hostname in selected:
        t0 = time.monotonic()
        target = _resolve_connectivity_target(hostname, snap)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(target, 22), timeout=5.0)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            ms = round((time.monotonic() - t0) * 1000, 1)
            results.append({"hostname": hostname, "status": "ok", "ms": ms, "target": target})
        except asyncio.TimeoutError:
            results.append({"hostname": hostname, "status": "timeout", "ms": None, "target": target})
        except Exception as e:
            results.append({"hostname": hostname, "status": "error",
                            "error": str(e)[:60], "ms": None, "target": target})
        ok = sum(1 for r in results if r["status"] == "ok")
        INFERENCE_FILE.write_text(json.dumps({
            "mode": "connectivity", "status": "checking",
            "results": results, "total": total,
            "message": f"Checked {len(results)}/{total}…",
        }))

    ok = sum(1 for r in results if r["status"] == "ok")
    INFERENCE_FILE.write_text(json.dumps({
        "mode": "connectivity", "status": "done",
        "results": results, "total": total,
        "message": (f"All {total} reachable ✓" if ok == total
                    else f"{ok}/{total} reachable"),
    }))



# ── Chat proxy (forward to inference API) ──────────────────────────────────────
def _get_inference_api_url() -> Optional[str]:
    """Get the inference API URL.

    launch_api.sh always runs on the dashboard machine (localhost), so we only
    need the port from the config — never a remote IP.
    """
    config_path = Path(__file__).parent.parent / "configs" / "inference" / "cluster_config_inference.yaml"
    api_port = 8080  # default from cluster_config_inference.yaml
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            api_port = config.get("web_interface", {}).get("api_port", api_port)
        except Exception as e:
            logger.warning(f"Could not read inference config for api_port: {e}")
    return f"http://127.0.0.1:{api_port}"


@app.post("/chat")
async def chat_proxy(request: Request):
    """Proxy chat requests to the inference API server as a transparent SSE stream."""
    api_url = _get_inference_api_url()
    if not api_url:
        raise HTTPException(503, "Inference API server not configured or unreachable")

    chat_endpoint = f"{api_url}/chat"

    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    async def stream_from_api():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    chat_endpoint,
                    content=body,
                    headers={"content-type": content_type},
                ) as response:
                    if response.status_code != 200:
                        resp_text = (await response.aread()).decode(errors="ignore").strip()
                        error_msg = f"Inference API error: {response.status_code}"
                        if resp_text:
                            error_msg = f"{error_msg} - {resp_text[:180]}"
                        yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
                        return

                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': 'Inference API timeout', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Error proxying chat request: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        stream_from_api(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        # Replace any non-finite floats (NaN/Inf) so json.dumps stays valid JSON
        return {k: (None if isinstance(v, float) and not (v == v and v != float("inf") and v != float("-inf")) else v)
                for k, v in raw.items()}
    except Exception:
        return {}
