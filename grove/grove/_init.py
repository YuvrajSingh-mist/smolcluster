"""Initialization paths for grove.init()."""

import os


def _init_cluster(cluster: str, ws: int, timeout: float) -> None:
    import grove
    from .cluster import _discover_peers
    from .store.tcp_store import TCPStore
    from .group import Group
    from .comm import Communicator
    from .coordinator import CoordinatorServer, WorkerClient
    from ._types import TransportType, DEFAULT_BASE_PORT

    my_rank, peers, _zc = _discover_peers(cluster, ws, timeout)
    grove.rank = my_rank
    grove.world_size = ws
    coord_host = peers[0]["host"]
    store = TCPStore(
        rank=my_rank,
        world_size=ws,
        host=coord_host,
        port=DEFAULT_BASE_PORT - 199,
    )
    group = Group(my_rank, ws, store, TransportType.TCP)

    coord_port = DEFAULT_BASE_PORT - 198
    if my_rank == 0:
        addr_map = {r: peers[r]["host"] for r in range(ws)}
        grove._coordinator = CoordinatorServer(coord_host, coord_port, ws, addr_map)
        script_path = os.environ.get("GROVE_SCRIPT")
        if script_path and os.path.exists(script_path):
            grove._coordinator.set_script(script_path)

    worker_client = WorkerClient(coord_host, coord_port, my_rank)
    grove._worker_client = worker_client
    grove._comm = Communicator(group, worker_client)
    _init_packer()


def _init_p2p(cluster: str, ws: int, timeout: float) -> None:
    import grove
    from .transport.hybrid import HybridTransport
    from .group import Group
    from .comm import Communicator
    from .coordinator import P2PCoordinatorServer, P2PWorkerClient
    from .control import MsgType, encode_ctrl, decode_ctrl

    is_coord = os.environ.get("GROVE_IS_COORDINATOR") == "1"
    uid = os.environ.get("GROVE_P2P_UID", "")
    script = os.environ.get("GROVE_SCRIPT", "")
    script_name = os.path.basename(script) if script else ""

    transport = HybridTransport.create_standalone(
        cluster, ws,
        is_coordinator=is_coord,
        uid=uid,
        script_name=script_name,
    )
    grove.rank = transport.rank
    grove.world_size = ws

    if ws > 1 and not os.environ.get("GROVE_NO_WIFI"):
        transport._try_wifi_upgrade()

    if ws > 1:
        _distribute_script(grove, transport, script, ws, encode_ctrl, decode_ctrl, MsgType)

    if grove.rank == 0:
        grove._coordinator = P2PCoordinatorServer(ws, transport)

    worker = P2PWorkerClient(transport, grove.rank)
    grove._worker_client = worker

    if grove.rank == 0 and grove._coordinator is not None:
        grove._coordinator.set_local_worker(worker)

    group = Group(grove.rank, ws, store=None, transport=transport)
    grove._comm = Communicator(group, worker)
    _init_packer()

    if not transport._upgraded:
        transport.start_keepalive()


def _distribute_script(grove, transport, script, ws, encode_ctrl, decode_ctrl, MsgType):
    import json
    if grove.rank == 0 and script and os.path.exists(script):
        with open(script) as f:
            content = f.read()
        # include sys.argv[1:] so workers can reconstruct the exact argv
        import sys as _sys
        msg = encode_ctrl(MsgType.SCRIPT_RESPONSE, {
            "name": os.path.basename(script),
            "content": content,
            "argv": json.dumps(_sys.argv[1:]),
        })
        for r in range(1, ws):
            transport.send_raw(r, msg)
    elif grove.rank != 0:
        raw = transport.recv_ctrl(0)
        msg_type, payload = decode_ctrl(raw)
        if msg_type != MsgType.SCRIPT_RESPONSE:
            raise ValueError(f"Expected script, got {msg_type}")
        grove._received_script = (payload["name"], payload["content"])
        grove._received_argv = json.loads(payload.get("argv", "[]"))


def _init_packer() -> None:
    import grove
    from .mlx_comm import _GradientPacker
    grove._packer = _GradientPacker()
