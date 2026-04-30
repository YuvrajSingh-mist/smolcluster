"""Peer discovery for smolcluster.

Wraps grove's peer discovery so smolcluster algorithms can find each other
without manual IP configuration.

Mac (default): zeroconf mDNS over WiFi/Ethernet
Mac (AWDL):    set SMOLCLUSTER_TRANSPORT=awdl — no router needed, direct peer WiFi
Other OS:      zeroconf mDNS

Usage:
    from smolcluster.discovery import discover

    my_rank, peers, zc = discover(cluster="my-run", world_size=2)
    coord_ip = peers[0]["host"]
    # ... start training ...
    zc.close()   # call when training is fully started, not before
"""

import os
import platform
import socket
import sys


def discover(
    cluster: str,
    world_size: int,
    timeout: float = 60.0,
) -> tuple[int, dict[int, dict], object]:
    """Block until all world_size nodes found.

    Returns (my_rank, peers, zc) where:
        my_rank  — this node's rank (0 = coordinator)
        peers    — {rank: {"host": ip, "hostname": name}}
        zc       — Zeroconf handle (or no-op), call zc.close() after training starts
    """
    # When running under grove the P2P channel is already established —
    # skip mDNS entirely and exchange IPs through grove's existing connection.
    try:
        import grove as _grove
        if _grove.world_size > 1:
            peer_map = _grove.peers(timeout=timeout)
            my_rank = _grove.rank
            # Return a no-op zc handle so caller code stays uniform
            class _Noop:
                def close(self): pass
            return my_rank, {int(r): v for r, v in peer_map.items()}, _Noop()
    except ImportError:
        pass

    transport = os.environ.get("SMOLCLUSTER_TRANSPORT", "mdns").lower()

    if transport == "awdl" and platform.system() == "Darwin":
        return _discover_awdl(cluster, world_size, timeout)
    return _discover_mdns(cluster, world_size, timeout)


def _grove_path() -> str:
    return os.path.join(os.path.dirname(__file__), "../../../grove")


def _discover_mdns(cluster: str, world_size: int, timeout: float):
    sys.path.insert(0, _grove_path())
    try:
        from grove.cluster import _discover_peers
    except ImportError:
        raise ImportError(
            "grove not found. Expected at <repo>/grove/. "
            "Run: cd grove && pip install -e ."
        )

    os.environ.setdefault("GROVE_IS_COORDINATOR", "0")
    my_rank, peers, zc = _discover_peers(cluster, world_size, timeout)
    # zc intentionally returned — caller must keep it alive until training
    # starts so slower nodes can still discover this node via mDNS
    return my_rank, peers, zc


def _discover_awdl(cluster: str, world_size: int, timeout: float):
    sys.path.insert(0, _grove_path())
    try:
        from grove.transport.p2p import discover_p2p_clusters
        from grove._utils import get_local_ip
    except ImportError:
        raise ImportError("grove not found — required for AWDL discovery.")

    clusters = discover_p2p_clusters(timeout=timeout)
    match = next((c for c in clusters if c.get("name") == cluster), None)
    if match is None:
        raise TimeoutError(
            f"AWDL: cluster '{cluster}' not found within {timeout}s. "
            "Is the coordinator running grove start?"
        )

    is_coord = os.environ.get("GROVE_IS_COORDINATOR") == "1"
    my_rank = 0 if is_coord else 1
    coord_ip = match.get("host", "")

    peers = {0: {"host": coord_ip, "hostname": match.get("hostname", "coordinator")}}
    if not is_coord:
        peers[my_rank] = {"host": get_local_ip(), "hostname": socket.gethostname()}

    # No zeroconf handle for AWDL — return a no-op so caller code is uniform
    class _Noop:
        def close(self): pass

    return my_rank, peers, _Noop()
