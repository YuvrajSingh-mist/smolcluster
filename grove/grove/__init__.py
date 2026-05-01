"""Grove: Zero-config distributed ML for Apple Silicon."""

__version__ = "0.1.0"

rank: int = 0
world_size: int = 1

_comm = None  # stub — collective ops removed; cli.py checks this before use
_coordinator = None
_worker_client = None
_received_script: tuple[str, str] | None = None
_received_argv: list = []


class World:
    def rank(self) -> int:
        import grove
        return grove.rank

    def size(self) -> int:
        import grove
        return grove.world_size


def init(
    cluster: str | None = None,
    world_size: int | None = None,
    timeout: float = 120.0,
    transport: str = "p2p",
) -> World:
    if cluster is not None and world_size is not None and world_size > 1:
        if transport == "p2p":
            _init_p2p(cluster, world_size, timeout)
        else:
            _init_cluster(cluster, world_size, timeout)
    return World()


def report(
    loss: float,
    step: int | None = None,
    *,
    grad_norm: float | None = None,
    tok_per_sec: float | None = None,
    tx_mbps: float | None = None,
    rx_mbps: float | None = None,
) -> None:
    if _worker_client is not None:
        _worker_client._loss = loss
        if step is not None:
            _worker_client._step = step
        if grad_norm is not None:
            _worker_client._grad_norm = grad_norm
        if tok_per_sec is not None:
            _worker_client._tok_per_sec = tok_per_sec
        if tx_mbps is not None:
            _worker_client._tx_mbps = tx_mbps
        if rx_mbps is not None:
            _worker_client._rx_mbps = rx_mbps
    if _coordinator is not None:
        with _coordinator._lock:
            _coordinator._loss[rank] = loss
            if step is not None:
                _coordinator._step_counts[rank] = step
            if grad_norm is not None:
                _coordinator._grad_norm[rank] = grad_norm
            if tok_per_sec is not None:
                _coordinator._tok_per_sec[rank] = tok_per_sec
            if tx_mbps is not None:
                _coordinator._tx_mbps[rank] = tx_mbps
            if rx_mbps is not None:
                _coordinator._rx_mbps[rank] = rx_mbps


def status(msg: str) -> None:
    if _worker_client is not None:
        _worker_client.set_status(msg)
    if _coordinator is not None:
        with _coordinator._lock:
            _coordinator._status_map[rank] = msg


def peers(timeout: float = 30.0) -> dict:
    import time as _time
    from ._utils import get_local_ip

    if world_size <= 1:
        return {0: {"host": get_local_ip(), "hostname": __import__("socket").gethostname()}}

    deadline = _time.monotonic() + timeout

    if _coordinator is not None:
        while _time.monotonic() < deadline:
            with _coordinator._lock:
                ips = dict(_coordinator._peer_ips)
                hostnames = dict(_coordinator._hostnames)
            hostnames.setdefault(0, __import__("socket").gethostname())
            if len(ips) >= world_size:
                return {r: {"host": ips[r], "hostname": hostnames.get(r, f"rank-{r}")}
                        for r in range(world_size)}
            _time.sleep(0.2)
        raise TimeoutError(f"grove.peers(): only got IPs for {len(ips)}/{world_size} nodes within {timeout}s")

    if _worker_client is not None:
        while _time.monotonic() < deadline:
            stats = _worker_client.get_cluster_stats()
            if stats and "ips" in stats and len(stats["ips"]) >= world_size:
                ips = stats["ips"]
                hostnames = stats.get("hostnames", {})
                return {int(r): {"host": ips[r], "hostname": hostnames.get(r, f"rank-{r}")}
                        for r in ips}
            _time.sleep(0.2)
        stats = _worker_client.get_cluster_stats() or {}
        got = len(stats.get("ips", {}))
        raise TimeoutError(f"grove.peers(): only got IPs for {got}/{world_size} nodes within {timeout}s")

    return {}


def is_available() -> bool:
    return world_size > 1


from ._init import _init_cluster, _init_p2p  # noqa: E402
