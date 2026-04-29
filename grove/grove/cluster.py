"""Cluster discovery and formation via Bonjour."""

import os
import socket
import threading
import time
import uuid
from ._types import DEFAULT_BASE_PORT
from ._utils import get_logger, get_local_ip

log = get_logger("cluster")

_CLUSTER_SERVICE = "_grove._tcp.local."
_COORD_PORT = DEFAULT_BASE_PORT - 200


def generate_uid() -> str:
    return uuid.uuid4().hex[:4]


def advertise_cluster(
    name: str,
    uid: str,
    expected: int,
    mode: str,
    script: str,
    host: str | None = None,
) -> tuple:
    from zeroconf import Zeroconf, ServiceInfo

    host = host or get_local_ip()
    zc = Zeroconf()
    service_name = f"grove-{uid}.{_CLUSTER_SERVICE}"
    info = ServiceInfo(
        _CLUSTER_SERVICE, service_name,
        addresses=[socket.inet_aton(host)],
        port=_COORD_PORT + (os.getpid() % 1000),
        properties={
            b"name": name.encode(),
            b"uid": uid.encode(),
            b"hostname": socket.gethostname().encode(),
            b"host": host.encode(),
            b"expected": str(expected).encode(),
            b"current": b"1",
            b"script": os.path.basename(script).encode(),
            b"mode": mode.encode(),
            b"started": str(time.time()).encode(),
        },
    )
    zc.register_service(info)
    log.info(f"Advertising cluster '{name}' ({uid}) on {host}")
    return zc, info


def update_cluster_count(zc, info, count: int) -> None:
    from zeroconf import ServiceInfo
    props = dict(info.properties)
    props[b"current"] = str(count).encode()
    info.properties = props
    zc.update_service(info)


class LiveBrowser:
    def __init__(self) -> None:
        from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange

        self._clusters: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._zc = Zeroconf()

        def on_change(zeroconf, service_type, name, state_change):
            if state_change == ServiceStateChange.Added:
                sinfo = zeroconf.get_service_info(service_type, name)
                if sinfo is None:
                    return
                props = {k.decode(): v.decode() for k, v in sinfo.properties.items()}
                uid = props.get("uid", "")
                if uid:
                    with self._lock:
                        self._clusters[uid] = props
            elif state_change == ServiceStateChange.Removed:
                sinfo = zeroconf.get_service_info(service_type, name)
                if sinfo:
                    props = {k.decode(): v.decode() for k, v in sinfo.properties.items()}
                    uid = props.get("uid", "")
                    with self._lock:
                        self._clusters.pop(uid, None)

        self._browser = ServiceBrowser(self._zc, _CLUSTER_SERVICE, handlers=[on_change])

    def get_clusters(self) -> list[dict]:
        with self._lock:
            return sorted(self._clusters.values(), key=lambda c: c.get("started", "0"))

    def close(self) -> None:
        self._zc.close()


def browse_clusters_live() -> LiveBrowser:
    return LiveBrowser()


def browse_clusters(timeout: float = 5.0) -> list[dict]:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange

    zc = Zeroconf()
    clusters: dict[str, dict] = {}
    lock = threading.Lock()

    def on_change(zeroconf, service_type, name, state_change):
        if state_change != ServiceStateChange.Added:
            return
        sinfo = zeroconf.get_service_info(service_type, name)
        if sinfo is None:
            return
        props = {k.decode(): v.decode() for k, v in sinfo.properties.items()}
        uid = props.get("uid", "")
        if uid:
            with lock:
                clusters[uid] = props

    ServiceBrowser(zc, _CLUSTER_SERVICE, handlers=[on_change])
    time.sleep(timeout)
    zc.close()

    with lock:
        return sorted(clusters.values(), key=lambda c: c.get("started", "0"))


def _discover_peers(cluster: str, world_size: int, timeout: float) -> tuple[int, dict]:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo, ServiceStateChange

    host = get_local_ip()
    pid = os.getpid()
    is_coord = os.environ.get("GROVE_IS_COORDINATOR") == "1"
    peer_id = f"{'0' if is_coord else '1'}-{host}:{pid}"
    zc = Zeroconf()

    service_name = f"grove-{cluster}-{pid}.{_CLUSTER_SERVICE}"
    info = ServiceInfo(
        _CLUSTER_SERVICE, service_name,
        addresses=[socket.inet_aton(host)],
        port=_COORD_PORT + (pid % 1000),
        properties={
            b"cluster": cluster.encode(),
            b"hostname": socket.gethostname().encode(),
            b"host": host.encode(),
            b"peer_id": peer_id.encode(),
        },
    )
    zc.register_service(info)
    log.info(f"Advertising on cluster '{cluster}' as {socket.gethostname()} ({peer_id})")

    discovered = {peer_id: {"host": host, "hostname": socket.gethostname(), "peer_id": peer_id}}
    lock = threading.Lock()
    all_found = threading.Event()

    def on_change(zeroconf, service_type, name, state_change):
        if state_change != ServiceStateChange.Added:
            return
        sinfo = zeroconf.get_service_info(service_type, name)
        if sinfo is None:
            return
        props = {k.decode(): v.decode() for k, v in sinfo.properties.items()}
        if props.get("cluster") != cluster:
            return
        pid_key = props.get("peer_id", "")
        with lock:
            if pid_key and pid_key not in discovered:
                discovered[pid_key] = {
                    "host": props.get("host", ""),
                    "hostname": props.get("hostname", ""),
                    "peer_id": pid_key,
                }
                log.info(f"Discovered: {props.get('hostname')} ({pid_key}) "
                         f"[{len(discovered)}/{world_size}]")
                if len(discovered) >= world_size:
                    all_found.set()

    ServiceBrowser(zc, _CLUSTER_SERVICE, handlers=[on_change])

    if not all_found.wait(timeout):
        zc.unregister_all_services()
        zc.close()
        raise TimeoutError(f"Only found {len(discovered)}/{world_size} peers within {timeout}s")

    with lock:
        sorted_ids = sorted(discovered.keys())
    my_rank = sorted_ids.index(peer_id)
    roster = ", ".join(f"{r}={discovered[pid]['hostname']}" for r, pid in enumerate(sorted_ids))
    log.info(f"Rank assignment: {my_rank}/{world_size} ({roster})")

    return my_rank, {r: discovered[pid] for r, pid in enumerate(sorted_ids)}, zc
