"""SSH config parsing and node-lookup helpers."""
import logging
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_ssh_config() -> dict:
    """
    Parse ~/.ssh/config and return
    {key: {alias, user, hostname}} where key may be alias, HostName, or variants.
    Skips wildcard/glob Host entries.
    """
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        return {}
    result: dict = {}
    current_host: Optional[str] = None
    current: dict = {}

    def _flush():
        if current_host and "*" not in current_host and "?" not in current_host:
            alias     = current_host.split()[0]
            host_name = current.get("Hostname", "")
            entry = {"alias": alias, "user": current.get("User", ""), "hostname": host_name}
            keys = {alias}
            if host_name:
                keys.add(host_name)
                keys.add(host_name.removesuffix(".local") if host_name.endswith(".local")
                         else f"{host_name}.local")
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


# Parsed once at import time — all lookups use this dict.
_SSH_CONFIG: dict = parse_ssh_config()


def _lookup_ssh_entry(hostname: str, node_ip: str) -> dict:
    """Resolve a discovered node to an SSH config entry (priority: IP, FQDN, bare name)."""
    for key in (node_ip, f"{hostname}.local", hostname):
        if key and key in _SSH_CONFIG:
            return _SSH_CONFIG[key]
    # Heuristic for common Jetson cluster naming (jetson-nano1 → jetson, …2 → jetson2)
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
    """Return the SSH config alias for the server, or the hostname itself."""
    candidates: list[str] = []
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=3).stdout
        for m in re.findall(r'\binet\s+(\d+\.\d+\.\d+\.\d+)', out):
            if not m.startswith("127.") and not m.startswith("169.254."):
                candidates.append(m)
    except Exception:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.strip()
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
        host_name          = (entry.get("hostname") or "").strip()
        host_name_no_local = host_name.removesuffix(".local")
        if alias == server_hostname or host_name in local_ips or host_name_no_local == server_hostname:
            continue
        nodes[alias] = {
            "hostname": alias, "alias": alias, "ip": host_name,
            "port": 22, "os": "", "os_version": "", "machine": "",
            "role": "available", "source": "ssh_config",
        }
    return nodes


def local_node_metadata() -> dict:
    return {
        "os":         platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "machine":    platform.machine(),
    }
