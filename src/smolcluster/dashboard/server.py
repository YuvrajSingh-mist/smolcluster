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
"""

import asyncio
import json
import logging
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from smolcluster.dashboard.node_manager import NodeManager
from smolcluster.utils.discovery import NodeDiscovery, register_node

logger = logging.getLogger(__name__)

FRONTEND_DIR   = Path(__file__).parent / "frontend"
METRICS_FILE   = Path("/tmp/smolcluster_metrics.json")
INFERENCE_FILE = Path("/tmp/smolcluster_inference.json")


# ── SSH config parsing ─────────────────────────────────────────────────────────
def parse_ssh_config() -> dict:
    """
    Parse ~/.ssh/config and return {ip_or_hostname: {'alias': 'mini2', 'user': 'yuvrajsingh2'}}.
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
            key = current.get("hostname", current_host)
            result[key] = {"alias": current_host, "user": current.get("user", "")}

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
            current["hostname"] = v
        elif k == "user":
            current["user"] = v
    _flush()
    return result


_SSH_CONFIG:  dict = parse_ssh_config()   # ip/hostname → {alias, user}
_ssh_aliases: dict = {}                   # mDNS hostname → SSH alias e.g. "mini2"


def _get_local_ip() -> str:
    """Best-effort: get this machine's LAN IP."""
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
    local_ip = _get_local_ip()
    if local_ip and local_ip in _SSH_CONFIG:
        return _SSH_CONFIG[local_ip]["alias"]
    if server_hostname in _SSH_CONFIG:
        return _SSH_CONFIG[server_hostname]["alias"]
    return server_hostname

# ── App state ─────────────────────────────────────────────────────────────────
discovery:        NodeDiscovery
node_manager:     NodeManager
_zc               = None
_server_hostname: str = ""
_loop:            asyncio.AbstractEventLoop = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global discovery, node_manager, _zc, _server_hostname, _loop
    _loop = asyncio.get_running_loop()
    discovery    = NodeDiscovery(on_change=_on_node_change)
    node_manager = NodeManager()
    _server_hostname = socket.gethostname().removesuffix(".local")
    _zc = await asyncio.to_thread(register_node, 8080, "server", _server_hostname)
    logger.info(f"[dashboard] http://{_server_hostname}.local:8080")
    yield
    await node_manager.stop_all()
    discovery.close()
    if _zc:
        await asyncio.to_thread(_zc.close)


app = FastAPI(title="smolcluster Dashboard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Auto-probe usernames ───────────────────────────────────────────────────────
_probed:   dict = {}  # hostname → SSH username
_node_os:  dict = {}  # hostname → {os, os_version, machine}

def _on_node_change():
    for hostname, info in discovery.snapshot().items():
        if hostname not in _probed:
            _probed[hostname] = None
            if _loop and _loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    _probe_and_store(hostname, info), _loop)

async def _probe_and_store(hostname: str, info: dict):
    import re

    # 1. Check ~/.ssh/config by IP first (fastest, most reliable)
    node_ip = discovery.snapshot().get(hostname, {}).get("ip", "")
    ssh_entry = _SSH_CONFIG.get(node_ip, {})
    if ssh_entry:
        _ssh_aliases[hostname] = ssh_entry["alias"]
        _probed[hostname] = ssh_entry.get("user", "")
        logger.info(f"[dashboard] {hostname} → SSH alias '{ssh_entry['alias']}' from ~/.ssh/config")
        target = ssh_entry["alias"]
    else:
        # 2. Fall back to SSH probe (whoami)
        m = re.search(r'macmini(\d+)', hostname, re.IGNORECASE)
        guess = f"yuvrajsingh{m[1]}" if m else ""
        target = None
        for attempt in ([guess] if guess else []) + [""]:
            user = await NodeManager.probe_username(hostname, attempt)
            if user:
                _probed[hostname] = user
                target = _build_ssh_target(attempt, hostname)
                break
        if _probed.get(hostname) is None:
            _probed[hostname] = ""

    # 3. Probe OS info via SSH — await directly so it's ready before next SSE tick
    if target and hostname not in _node_os:
        await _probe_os(hostname, target)


async def _probe_os(hostname: str, target: str) -> None:
    """SSH-probe OS info: uname -srm + sw_vers -productVersion (macOS)."""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           target,
           "uname -s; uname -r; uname -m; sw_vers -productVersion 2>/dev/null || true"]
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True),
            timeout=12.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            os_name    = lines[0] if len(lines) > 0 else ""
            kernel_ver = lines[1] if len(lines) > 1 else ""
            machine    = lines[2] if len(lines) > 2 else ""
            # sw_vers gives the human macOS version (e.g. "15.4.1"); prefer it
            os_version = lines[3] if len(lines) > 3 else kernel_ver
            if os_name:
                _node_os[hostname] = {
                    "os": os_name, "os_version": os_version, "machine": machine,
                }
                logger.info(f"[dashboard] {hostname} OS: {os_name} {os_version} {machine}")
        else:
            logger.warning(f"[dashboard] OS probe {hostname}: rc={result.returncode} stderr={result.stderr[:80]}")
    except Exception as e:
        logger.warning(f"[dashboard] OS probe {hostname}: {e}")


# ── Request models ─────────────────────────────────────────────────────────────
class SelectRequest(BaseModel):
    ssh_user: str = ""
    rank: Optional[int] = None

class StartRequest(BaseModel):
    algorithm: str = "syncps"

class InferenceLaunchRequest(BaseModel):
    algorithm: str = "syncps"
    server_hostname: str = ""   # which selected node is the server/rank-0


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/nodes")
async def get_nodes():
    return {
        "discovered":  discovery.snapshot(),
        "selected":    node_manager.snapshot_selected(),
        "running":     node_manager.snapshot_processes(),
        "usernames":   dict(_probed),
        "ssh_aliases": dict(_ssh_aliases),
        "node_os":     dict(_node_os),
    }


@app.get("/api/nodes/{hostname}/probe")
async def probe_node(hostname: str, ssh_user: str = ""):
    user = await NodeManager.probe_username(hostname, ssh_user)
    if user is None:
        raise HTTPException(502, "Unreachable or SSH key not set up")
    _probed[hostname] = user
    return {"username": user}


@app.post("/api/nodes/{hostname}/select")
async def select_node(hostname: str, req: SelectRequest):
    rank = await node_manager.select(hostname, req.ssh_user, req.rank)
    return {"status": "selected", "rank": rank}


@app.post("/api/nodes/{hostname}/deselect")
async def deselect_node(hostname: str):
    await node_manager.deselect(hostname)
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
async def stop_training():
    await node_manager.stop_training()
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
    await node_manager.stop_training()
    INFERENCE_FILE.unlink(missing_ok=True)
    return {"status": "stopped"}


INFER_CONFIG_FILE = (Path(__file__).parent.parent /
                     "configs" / "inference" / "cluster_config_inference.yaml")
INFER_SCRIPT_FILE = (Path(__file__).parent.parent.parent.parent /
                     "scripts" / "inference" / "launch_inference.sh")


@app.post("/api/inference/launch")
async def launch_inference_script(req: InferenceLaunchRequest):
    """Write cluster_config_inference.yaml and run launch_inference.sh."""
    if not node_manager.selected:
        raise HTTPException(400, "No nodes selected")

    algorithm = req.algorithm
    snap      = discovery.snapshot()

    # Build nodes_info from selected nodes only — local machine is never included
    # (scripts run locally and rsync TO remote nodes; including self breaks rsync)
    nodes_info: dict = {}
    for hostname, sel in node_manager.selected.items():
        node_ip = snap.get(hostname, {}).get("ip", "")
        # Alias = SSH Host entry (e.g. "mini2") — look up by IP in ~/.ssh/config,
        # fall back to cached _ssh_aliases, then bare mDNS hostname.
        # Never use ssh_user (username) as the alias — they're different things.
        alias = (_SSH_CONFIG.get(node_ip, {}).get("alias")
                 or _ssh_aliases.get(hostname)
                 or hostname)
        user  = _probed.get(hostname, "")
        nodes_info[hostname] = {
            "ssh_alias": alias,
            "user":      user,
            "rank":      sel["rank"],
            "ip":        node_ip,
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


@app.post("/api/connectivity/check")
async def connectivity_check():
    """TCP port-22 probe — works without SSH keys."""
    selected = node_manager.snapshot_selected()
    if not selected:
        raise HTTPException(400, "No nodes selected")
    asyncio.create_task(_run_tcp_checks(selected))
    return {"status": "checking"}


@app.get("/api/training")
async def get_training():
    return _read_json(METRICS_FILE)


# ── SSE: state (1 Hz) ──────────────────────────────────────────────────────────
@app.get("/api/events")
async def sse_events():
    async def gen():
        while True:
            payload = json.dumps({
                "nodes": {
                    "discovered":  discovery.snapshot(),
                    "selected":    node_manager.snapshot_selected(),
                    "running":     node_manager.snapshot_processes(),
                    "usernames":   dict(_probed),
                    "ssh_aliases": dict(_ssh_aliases),
                    "node_os":     dict(_node_os),
                },
                "training":     _read_json(METRICS_FILE),
                "connectivity": _read_json(INFERENCE_FILE),
            })
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── SSE: logs (3 Hz) ───────────────────────────────────────────────────────────
@app.get("/api/logs")
async def sse_logs():
    async def gen():
        last_seq = 0
        while True:
            lines = node_manager.logs_since(last_seq)
            if lines:
                last_seq = lines[-1]["seq"]
                yield f"data: {json.dumps(lines)}\n\n"
            await asyncio.sleep(0.35)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── TCP connectivity check (background task) ───────────────────────────────────
async def _run_tcp_checks(selected: dict):
    total = len(selected)
    results = []
    INFERENCE_FILE.write_text(json.dumps({
        "mode": "connectivity", "status": "checking",
        "results": [], "total": total,
        "message": f"Checking {total} node(s)…",
    }))
    for hostname in selected:
        t0 = time.monotonic()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(f"{hostname}.local", 22), timeout=5.0)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            ms = round((time.monotonic() - t0) * 1000, 1)
            results.append({"hostname": hostname, "status": "ok", "ms": ms})
        except asyncio.TimeoutError:
            results.append({"hostname": hostname, "status": "timeout", "ms": None})
        except Exception as e:
            results.append({"hostname": hostname, "status": "error",
                            "error": str(e)[:60], "ms": None})
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
