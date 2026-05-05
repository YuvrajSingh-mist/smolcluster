"""Node metadata collection and background refresh."""
import json
import logging
import re
import socket
import subprocess

from . import ctx
from .ssh_config import _get_server_alias, _lookup_ssh_entry, local_node_metadata
from dashboard.node_manager import NodeManager

logger = logging.getLogger(__name__)


def collect_local_ips() -> set:
    """Enumerate all non-loopback IPs of this machine."""
    ips: set = set()
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=3).stdout
        for m in re.findall(r'\binet\s+(\d+\.\d+\.\d+\.\d+)', out):
            if not m.startswith("127.") and not m.startswith("169.254."):
                ips.add(m)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2,
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


async def refresh_node_metadata(hostname: str, ssh_hint: str = "") -> None:
    from .helpers import canonicalize_node_hostname  # local import avoids load-time cycle
    canonical = canonicalize_node_hostname(hostname)
    if not canonical:
        return

    if canonical == ctx.server_hostname:
        ctx.node_os[canonical] = local_node_metadata()
        return

    node_ip   = ctx.static_nodes.get(canonical, {}).get("ip", "")
    ssh_entry = _lookup_ssh_entry(canonical, node_ip)
    target    = ssh_hint or ssh_entry.get("alias") or ctx.ssh_aliases.get(canonical) or canonical
    info      = await NodeManager.probe_metadata(canonical, target)
    if not info:
        return

    username = (info.get("username") or "").strip()
    if username:
        ctx.probed[canonical] = username
        await ctx.redis.hset("smolcluster:probed", canonical, username)

    os_info = {
        "os":         (info.get("os") or "").strip(),
        "os_version": (info.get("os_version") or "").strip(),
        "machine":    (info.get("machine") or "").strip(),
    }
    if any(os_info.values()):
        ctx.node_os[canonical] = os_info
        await ctx.redis.hset("smolcluster:node_os", canonical, json.dumps(os_info))


async def prime_node_metadata() -> None:
    import asyncio
    await refresh_node_metadata(ctx.server_hostname)
    tasks = [
        asyncio.create_task(refresh_node_metadata(h))
        for h in ctx.static_nodes
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
