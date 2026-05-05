"""Misc helpers used across route handlers and broadcasters."""
import json
import logging
import platform
import re
import yaml
from pathlib import Path
from typing import Optional

from . import ctx
from .ssh_config import (
    _SSH_CONFIG, _get_local_ip, _get_server_alias, _lookup_ssh_entry,
    local_node_metadata,
)

logger = logging.getLogger(__name__)


def _self_node() -> dict:
    """Build the node entry for the local (server) machine."""
    alias = _get_server_alias(ctx.server_hostname)
    return {
        "hostname":   ctx.server_hostname,
        "alias":      alias,
        "ip":         _get_local_ip() or "127.0.0.1",
        "port":       9090,
        "os":         ctx.node_os.get(ctx.server_hostname, {}).get("os", platform.system()),
        "os_version": ctx.node_os.get(ctx.server_hostname, {}).get(
            "os_version", platform.mac_ver()[0] or platform.release()
        ),
        "machine":    ctx.node_os.get(ctx.server_hostname, {}).get("machine", platform.machine()),
        "role":       "server",
        "source":     "local",
    }


def _ssh_aliases_snapshot() -> dict:
    aliases = dict(ctx.ssh_aliases)
    aliases[ctx.server_hostname] = _get_server_alias(ctx.server_hostname)
    return aliases


def _looks_like_server_session(session: str) -> bool:
    session = (session or "").strip().lower()
    return bool(re.search(r"(^|[-_])server($|[-_])", session))


def canonicalize_node_hostname(hostname: str) -> str:
    name = (hostname or "").strip().removesuffix(".local")
    if not name:
        return name
    if name == ctx.server_hostname:
        return name
    if name in ctx.static_nodes:
        return name
    server_alias = (_get_server_alias(ctx.server_hostname) or "").removesuffix(".local")
    if name == server_alias:
        return ctx.server_hostname
    ssh_entry = _SSH_CONFIG.get(name)
    alias = (ssh_entry or {}).get("alias", "")
    if alias in ctx.static_nodes:
        return alias
    return name


def canonicalize_log_hostname(raw_hostname: str, session: str = "") -> str:
    """Resolve log host labels back to the dashboard's canonical node hostname."""
    hostname = (raw_hostname or "").strip().removesuffix(".local")
    if not hostname:
        return ctx.server_hostname if _looks_like_server_session(session) else "unknown"
    if hostname == ctx.server_hostname:
        return hostname
    server_alias = (_get_server_alias(ctx.server_hostname) or "").removesuffix(".local")
    if _looks_like_server_session(session) and hostname == server_alias:
        return ctx.server_hostname
    known = {
        ctx.server_hostname,
        *ctx.static_nodes.keys(),
        *(ctx.node_manager.snapshot_selected().keys() if ctx.node_manager else []),
        *(ctx.node_manager.snapshot_processes().keys() if ctx.node_manager else []),
    }
    if hostname in known:
        return hostname
    alias_matches = [
        canonical for canonical, alias in _ssh_aliases_snapshot().items()
        if (alias or "").strip().removesuffix(".local") == hostname
    ]
    if len(alias_matches) == 1:
        return alias_matches[0]
    if len(alias_matches) > 1 and _looks_like_server_session(session) and ctx.server_hostname in alias_matches:
        return ctx.server_hostname
    return hostname


def build_nodes_info(snap: dict) -> dict:
    """Build nodes_info dict from currently selected nodes (used by launch routes)."""
    nodes_info: dict = {}
    for hostname, sel in ctx.node_manager.selected.items():
        if hostname == ctx.server_hostname:
            node_ip     = _get_local_ip() or _self_node().get("ip", "")
            ssh_entry   = _lookup_ssh_entry(hostname, node_ip)
            local_alias = _get_server_alias(hostname)
        else:
            node_ip     = snap.get(hostname, {}).get("ip", "")
            ssh_entry   = _lookup_ssh_entry(hostname, node_ip)
            local_alias = ""
        alias = (
            ssh_entry.get("alias")
            or local_alias
            or ctx.ssh_aliases.get(hostname)
            or hostname
        )
        preferred_ip = ssh_entry.get("hostname") or node_ip
        user = (
            ctx.probed.get(hostname)
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
    return nodes_info


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {
            k: (None if isinstance(v, float) and not (v == v and abs(v) != float("inf")) else v)
            for k, v in raw.items()
        }
    except Exception:
        return {}


def _get_inference_api_url() -> Optional[str]:
    """Return the inference API base URL, reading the port from cluster config."""
    from .paths import _REPO_ROOT
    config_path = _REPO_ROOT / "src" / "smolcluster" / "configs" / "inference" / "cluster_config_inference.yaml"
    api_port = 8080
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            api_port = config.get("web_interface", {}).get("api_port", api_port)
        except Exception as e:
            logger.warning(f"Could not read inference config for api_port: {e}")
    return f"http://127.0.0.1:{api_port}"
