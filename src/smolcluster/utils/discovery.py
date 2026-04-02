"""
Zero-config cluster node discovery via mDNS/Bonjour (zeroconf) with
SSH-config static seeding.

Usage on the dashboard / server node:
    discovery = NodeDiscovery()          # start browsing
    zc = register_node(8080, "server")   # advertise this machine

Usage on worker nodes (optional):
    zc = register_node(0, "available")   # just make this node discoverable
"""

import logging
import platform
import socket
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
from zeroconf._exceptions import NonUniqueNameException

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_smolcluster._tcp.local."

# Additional service types to passively discover — any Mac with Remote Login
# (SSH) enabled advertises _ssh._tcp.local. without running smolcluster.
_PASSIVE_SERVICE_TYPES = [
    "_ssh._tcp.local.",          # macOS Remote Login, Raspberry Pi
    "_sftp-ssh._tcp.local.",     # SFTP (often co-advertised)
]


# ── SSH-config static seeding ──────────────────────────────────────────────────

def _load_ssh_config_nodes() -> Dict[str, dict]:
    """
    Parse ~/.ssh/config and return a dict of hostname → node-dict for every
    non-wildcard Host entry that has a HostName set.  These entries are used to
    pre-seed NodeDiscovery so that nodes on different subnets (where mDNS
    multicast does not reach) are still visible in the dashboard.

    Returns: {alias: {hostname, ip, user, alias}}
    """
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        return {}

    nodes: Dict[str, dict] = {}
    current_host: Optional[str] = None
    current: dict = {}

    def _flush():
        if not current_host:
            return
        if "*" in current_host or "?" in current_host:
            return
        alias    = current_host.split()[0]
        hostname = current.get("Hostname", "")
        user     = current.get("User", "")
        if not hostname:
            return
        nodes[alias] = {
            "hostname": alias,          # short name used for display / SSH
            "ip":       hostname,       # HostName is the reachable address
            "user":     user,
            "alias":    alias,
            "port":     22,
            "os":       "",
            "os_version": "",
            "machine":  "",
            "role":     "available",
            "source":   "ssh_config",
        }

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
    return nodes


def _local_ips() -> set:
    """Return ALL non-loopback IPv4 addresses bound to this machine.
    Uses 'hostname -I' (Linux) which enumerates every interface, unlike
    getaddrinfo which only returns IPs the hostname resolves to.
    """
    import subprocess
    ips: set = set()
    # Primary: hostname -I lists every interface IP on Linux
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        for ip in out.split():
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.add(ip)
    except Exception:
        pass
    # Fallback: getaddrinfo + default-route probe
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips


def _tcp_reachable(host: str, port: int = 22, timeout: float = 3.0) -> bool:
    """Return True if TCP port is open on host (no auth needed)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _local_ip() -> str:
    """Best-effort: get the default-route IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def register_node(port: int, role: str = "available", hostname: Optional[str] = None) -> Zeroconf:
    """
    Advertise this machine on the local network via mDNS.

    Args:
        port:     Port the dashboard/worker listens on (use 0 if not applicable).
        role:     "server" | "worker" | "available"
        hostname: Override hostname (defaults to socket.gethostname()).

    Returns:
        Zeroconf instance — call .close() on shutdown to stop advertising.
    """
    if hostname is None:
        hostname = socket.gethostname()
    # Strip any existing .local suffix — zeroconf appends the service type itself
    hostname = hostname.removesuffix(".local")

    ip = _local_ip()
    zc = Zeroconf()

    info = ServiceInfo(
        SERVICE_TYPE,
        f"{hostname}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=max(port, 1),  # zeroconf requires port >= 1
        properties={
            "hostname": hostname,
            "role": role,
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
        },
    )
    try:
        zc.register_service(info)
        logger.info(f"[discovery] Registered {hostname} ({ip}) as role={role}")
    except NonUniqueNameException:
        # A stale mDNS record from a previous run is still live (TTL ~75s).
        # Safe to ignore — the old record will expire on its own and the browse
        # side works regardless.
        logger.warning(
            f"[discovery] Service {hostname} already registered on the network "
            f"(stale record from a previous run). Restart in ~75s to re-register."
        )
    return zc


class NodeDiscovery:
    """
    Discovers smolcluster peers via mDNS and via ~/.ssh/config static seeding.

    mDNS works only within a subnet; SSH-config seeding ensures nodes on
    different subnets (e.g. a WireGuard / 10.10.x.x overlay) are always
    visible without needing multicast to reach them.

    Attributes:
        nodes: Dict[hostname, {ip, port, os, os_version, machine, role, hostname}]
    """

    def __init__(self, on_change: Optional[Callable] = None):
        """
        Args:
            on_change: Optional callback fired whenever the node list changes.
        """
        self.nodes: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._on_change = on_change
        self._zc = Zeroconf()
        # Browse for smolcluster nodes AND any SSH-advertising machine (no setup needed)
        all_types = [SERVICE_TYPE] + _PASSIVE_SERVICE_TYPES
        self._browser = ServiceBrowser(self._zc, all_types, self)

        # Seed from ~/.ssh/config immediately so cross-subnet nodes are visible
        # even when mDNS multicast cannot reach them.
        self._seed_from_ssh_config()

    def _seed_from_ssh_config(self) -> None:
        """
        Add nodes from ~/.ssh/config to the discovery table.
        Runs a background thread to TCP-probe each node. Nodes that pass
        show up right away in the snapshot; an mDNS event later will upgrade
        them with richer metadata.
        """
        ssh_nodes = _load_ssh_config_nodes()
        if not ssh_nodes:
            return

        # Add all configured nodes to the table immediately (unprobed state)
        local_hostname = socket.gethostname().removesuffix(".local")
        local_ip_set = _local_ips()
        with self._lock:
            for alias, node in ssh_nodes.items():
                # Skip if the SSH alias matches our hostname OR if the
                # HostName IP is one of our own interface addresses.
                if node["hostname"] == local_hostname or node["ip"] in local_ip_set:
                    continue   # skip self
                if alias not in self.nodes:
                    self.nodes[alias] = dict(node)
                    logger.info(
                        f"[discovery] Seeded node from SSH config: "
                        f"{alias} ({node['ip']})"
                    )

        # NOTE: do NOT call _on_change here — the caller (e.g. server.py lifespan)
        # has not yet assigned the global `discovery` variable.  The background
        # probe thread calls _on_change per-node once TCP reachability is known.

        # Probe reachability in background — updates 'role' once confirmed
        threading.Thread(
            target=self._probe_ssh_nodes,
            args=(ssh_nodes, local_hostname),
            daemon=True,
            name="discovery-ssh-probe",
        ).start()

    def _probe_ssh_nodes(self, ssh_nodes: Dict[str, dict], local_hostname: str) -> None:
        """Background: TCP-probe each SSH-config node and mark reachable ones."""
        local_ip_set = _local_ips()
        for alias, node in ssh_nodes.items():
            if node["hostname"] == local_hostname or node["ip"] in local_ip_set:
                continue
            ip = node["ip"]
            reachable = _tcp_reachable(ip, port=22, timeout=3.0)
            with self._lock:
                entry = self.nodes.get(alias)
                if entry and entry.get("source") == "ssh_config":
                    entry["role"] = "available" if reachable else "unreachable"
            status = "reachable" if reachable else "unreachable"
            logger.info(f"[discovery] SSH-config probe: {alias} ({ip}) → {status}")
            if self._on_change:
                self._on_change()

    # ── zeroconf listener interface ────────────────────────────────────────────

    def add_service(self, zc: Zeroconf, svc_type: str, name: str) -> None:
        try:
            info = zc.get_service_info(svc_type, name)
        except Exception:
            return
        if not info:
            return
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in (info.properties or {}).items()
        }

        # Derive a clean short hostname suitable for SSH and display.
        if svc_type == SERVICE_TYPE:
            # smolcluster service: we embedded hostname in props explicitly
            hostname = props.get("hostname", "")
        elif info.server:
            # SSH/SFTP: info.server is the SRV target, e.g. "macmini2.local."
            hostname = info.server.rstrip(".").removesuffix(".local")
        else:
            raw_name = name.replace(f".{svc_type}", "").rstrip(".")
            hostname = props.get("hostname") or raw_name.split(".")[0]

        hostname = hostname.removesuffix(".local")
        if not hostname:
            return

        # Skip self — don't add the dashboard machine to the peer list
        _self_hostname = socket.gethostname().removesuffix(".local")
        _self_ips = _local_ips()
        if hostname == _self_hostname:
            return

        # Pick the best IP: prefer cluster/LAN (non-link-local) over 169.254.x.x
        ip = ""
        for addr in info.addresses:
            candidate = socket.inet_ntoa(addr)
            if not candidate.startswith("169.254."):
                ip = candidate
                break
        if not ip and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])

        # Also skip if the discovered IP is ours (catches alias mismatches)
        if ip in _self_ips:
            return

        with self._lock:
            existing = self.nodes.get(hostname, {})
            # smolcluster registration takes priority over passive SSH discovery
            if existing.get("role") in ("server", "worker") and svc_type != SERVICE_TYPE:
                return
            # Don't downgrade a good LAN IP to a link-local one
            existing_ip = existing.get("ip", "")
            if existing_ip and not existing_ip.startswith("169.254.") and ip.startswith("169.254."):
                ip = existing_ip
            node = {
                "hostname": hostname,
                "ip": ip,
                "port": info.port,
                "os": props.get("os", existing.get("os", _guess_os(props))),
                "os_version": props.get("os_version", existing.get("os_version", "")),
                "machine": props.get("machine", existing.get("machine", "")),
                "role": props.get("role", existing.get("role", "available")),
                # mDNS-discovered entries are not from ssh_config
                "source": "mdns",
            }
            self.nodes[hostname] = node

        logger.info(f"[discovery] Found node: {hostname} ({ip}) via {svc_type}")
        if self._on_change:
            self._on_change()

    def remove_service(self, zc: Zeroconf, svc_type: str, name: str) -> None:
        hostname = name.split(".")[0]
        with self._lock:
            self.nodes.pop(hostname, None)
        logger.info(f"[discovery] Lost node: {hostname}")
        if self._on_change:
            self._on_change()

    def update_service(self, zc: Zeroconf, svc_type: str, name: str) -> None:
        self.add_service(zc, svc_type, name)

    # ── public helpers ─────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self.nodes)

    def close(self) -> None:
        self._zc.close()


def _guess_os(props: dict) -> str:
    """Guess OS from SSH TXT record properties (best effort)."""
    # macOS SSH TXT records sometimes include platform hints
    for v in props.values():
        if isinstance(v, str):
            v_lo = v.lower()
            if "darwin" in v_lo or "mac" in v_lo:
                return "Darwin"
            if "linux" in v_lo:
                return "Linux"
    return ""
