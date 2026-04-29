"""P2P transport over AWDL via Swift helper sidecar."""

import os
import socket
import struct
import subprocess
import threading
import time
import numpy as np
from .base import Transport, Connection
from .._types import MAGIC
from .._utils import get_logger

log = get_logger("p2p")

_OP_SEND = 0x01
_OP_RECV = 0x02
_OP_DISCONNECT = 0x03
_HEADER_FMT = "<4sIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_CTRL_MAGIC = b"GCTL"


def _read_line(sock: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()


def _recvall(fd: int, n: int) -> bytes:
    parts = []
    remaining = n
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise ConnectionError("Control socket closed")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def discover_p2p_clusters(timeout: float = 10.0) -> list[dict]:
    from ..swift.compile import ensure_compiled
    helper_path = ensure_compiled()
    ctrl_path = f"/tmp/grove_discover_{os.getpid()}.sock"

    proc = subprocess.Popen(
        [str(helper_path), "discover", ctrl_path],
        stderr=subprocess.PIPE,
    )

    deadline = time.monotonic() + 30.0
    sock = None
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(ctrl_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)

    if sock is None:
        proc.terminate()
        return []

    _read_line(sock)

    clusters = []
    end_time = time.monotonic() + timeout
    sock.settimeout(1.0)

    while time.monotonic() < end_time:
        try:
            text = _read_line(sock)
            if not text:
                break
            if text.startswith("found "):
                parts = text.split(None, 4)
                if len(parts) >= 5:
                    clusters.append({
                        "name": parts[1],
                        "uid": parts[2],
                        "expected": parts[3],
                        "current": "1",
                        "script": parts[4],
                        "hostname": "awdl-peer",
                        "started": str(time.time()),
                    })
        except socket.timeout:
            continue

    sock.close()
    proc.terminate()
    proc.wait(timeout=3)
    if os.path.exists(ctrl_path):
        os.unlink(ctrl_path)

    return clusters


class P2PLiveBrowser:
    def __init__(self) -> None:
        from ..swift.compile import ensure_compiled
        self._helper_path = ensure_compiled()
        self._ctrl_path = f"/tmp/grove_discover_{os.getpid()}.sock"
        self._clusters: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._running = True

        self._proc = subprocess.Popen(
            [str(self._helper_path), "discover", self._ctrl_path],
            stderr=subprocess.PIPE,
        )

        self._sock = self._connect(self._ctrl_path)
        _read_line(self._sock)

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _connect(self, path: str) -> socket.socket:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(path)
                return sock
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)
        raise TimeoutError("Could not connect to Swift discovery helper")

    def _read_loop(self) -> None:
        self._sock.settimeout(1.0)
        while self._running:
            try:
                text = _read_line(self._sock)
                if not text:
                    return
                if text.startswith("found "):
                    parts = text.split(None, 4)
                    if len(parts) >= 5:
                        uid = parts[2]
                        with self._lock:
                            self._clusters[uid] = {
                                "name": parts[1],
                                "uid": uid,
                                "expected": parts[3],
                                "current": "1",
                                "script": parts[4],
                                "hostname": "awdl-peer",
                                "started": str(time.time()),
                            }
                elif text.startswith("lost "):
                    uid = text.split()[1]
                    with self._lock:
                        self._clusters.pop(uid, None)
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                break

    def get_clusters(self) -> list[dict]:
        with self._lock:
            return list(self._clusters.values())

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        self._proc.terminate()
        self._proc.wait(timeout=3)
        if os.path.exists(self._ctrl_path):
            os.unlink(self._ctrl_path)


class P2PConnection(Connection):
    def __init__(
        self,
        peer_rank: int,
        ctrl_sock: socket.socket,
        data_queue: "RecvQueue",
        ctrl_queue: "RecvQueue",
    ) -> None:
        self._peer_rank = peer_rank
        self._ctrl_sock = ctrl_sock
        self._data_queue = data_queue
        self._ctrl_queue = ctrl_queue
        self._send_lock = threading.Lock()

    def send(self, buf: np.ndarray) -> None:
        flat = buf.ravel()
        shape_bytes = struct.pack(f"<{buf.ndim}q", *buf.shape)
        array_header = struct.pack(_HEADER_FMT, MAGIC, flat.nbytes, buf.dtype.num, buf.ndim)
        payload = array_header + shape_bytes + bytes(memoryview(flat))
        relay_header = struct.pack("<BII", _OP_SEND, self._peer_rank, len(payload))
        with self._send_lock:
            self._ctrl_sock.sendall(relay_header + payload)

    def recv(self, buf: np.ndarray) -> None:
        payload = self._data_queue.get(self._peer_rank)
        magic, length, _dtype_num, ndim = struct.unpack(_HEADER_FMT, payload[:_HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"Bad magic in P2P recv: {magic!r}")
        offset = _HEADER_SIZE + ndim * 8
        result = np.frombuffer(payload[offset:offset + length], dtype=buf.dtype).reshape(buf.shape)
        np.copyto(buf, result)

    def send_raw(self, data: bytes) -> None:
        relay_header = struct.pack("<BII", _OP_SEND, self._peer_rank, len(data))
        with self._send_lock:
            self._ctrl_sock.sendall(relay_header + data)

    def recv_ctrl(self, timeout: float = 30.0) -> bytes:
        return self._ctrl_queue.get(self._peer_rank, timeout=timeout)

    def close(self) -> None:
        pass


class RecvQueue:
    def __init__(self) -> None:
        self._queues: dict[int, list[bytes]] = {}
        self._events: dict[int, threading.Event] = {}
        self._aborted: set[int] = set()
        self._lock = threading.Lock()

    def put(self, peer_rank: int, data: bytes) -> None:
        with self._lock:
            if peer_rank not in self._queues:
                self._queues[peer_rank] = []
                self._events[peer_rank] = threading.Event()
            self._queues[peer_rank].append(data)
            self._events[peer_rank].set()

    def abort(self, peer_rank: int) -> None:
        with self._lock:
            self._aborted.add(peer_rank)
            if peer_rank not in self._events:
                self._events[peer_rank] = threading.Event()
                self._queues[peer_rank] = []
            self._events[peer_rank].set()

    def get(self, peer_rank: int, timeout: float | None = None) -> bytes:
        with self._lock:
            if peer_rank not in self._events:
                self._events[peer_rank] = threading.Event()
                self._queues[peer_rank] = []

        self._events[peer_rank].wait(timeout)

        with self._lock:
            if peer_rank in self._aborted:
                raise ConnectionError(f"Peer {peer_rank} disconnected")
            if not self._queues[peer_rank]:
                raise TimeoutError(f"No data from peer {peer_rank}")
            data = self._queues[peer_rank].pop(0)
            if not self._queues[peer_rank]:
                self._events[peer_rank].clear()
        return data


class P2PTransport(Transport):
    def __init__(self, rank: int, world_size: int, store: "Store") -> None:
        from ..swift.compile import ensure_compiled

        self._rank = rank
        self._world_size = world_size
        self._connections: dict[int, P2PConnection] = {}
        self._data_queue = RecvQueue()
        self._ctrl_queue = RecvQueue()
        self._disconnected_peers: set[int] = set()
        self._keepalive_running = False

        helper_path = ensure_compiled()
        ctrl_path = f"/tmp/grove_p2p_{os.getpid()}.sock"
        self._ctrl_path = ctrl_path

        cluster_name = store.get("cluster_name").decode() if hasattr(store, "get") else "grove"
        log.info(f"Rank {rank}: starting P2P helper...")
        self._process = subprocess.Popen(
            [str(helper_path), "mesh", cluster_name, str(world_size), ctrl_path],
            stderr=subprocess.PIPE,
        )

        self._ctrl_sock = self._connect_helper(ctrl_path)
        self._ready_event = threading.Event()
        self._ready_line = ""
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._wait_ready()
        log.info(f"Rank {rank}: P2P helper ready")

        store.set(f"p2p_ready/{rank}", b"1")
        store.wait([f"p2p_ready/{i}" for i in range(world_size)])
        log.info(f"Rank {rank}: P2P transport ready")

    def _connect_helper(self, path: str, timeout: float = 30.0) -> socket.socket:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(path)
                return sock
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)
        raise TimeoutError("Could not connect to Swift P2P helper")

    def _wait_ready(self, timeout: float = 30.0) -> str:
        if not self._ready_event.wait(timeout):
            stderr = self._process.stderr.read() if self._process.stderr else b""
            raise ConnectionError(f"Swift helper died before ready: {stderr.decode()}")
        return self._ready_line

    def _read_loop(self) -> None:
        fd = self._ctrl_sock.fileno()
        try:
            while True:
                first = _recvall(fd, 1)
                if first[0] == ord('r'):
                    buf = first
                    while not buf.endswith(b"\n"):
                        buf += _recvall(fd, 1)
                    self._ready_line = buf.decode().strip()
                    self._ready_event.set()
                    continue
                rest = _recvall(fd, 8)
                op = first[0]
                peer_rank = struct.unpack("<I", rest[:4])[0]
                length = struct.unpack("<I", rest[4:8])[0]
                if op == _OP_DISCONNECT:
                    log.warning(f"Peer {peer_rank} disconnected (transport notification)")
                    self._disconnected_peers.add(peer_rank)
                    self._data_queue.abort(peer_rank)
                elif op == _OP_RECV and length > 0:
                    payload = _recvall(fd, length)
                    if len(payload) >= 4 and payload[:4] == MAGIC:
                        self._data_queue.put(peer_rank, payload)
                    else:
                        self._ctrl_queue.put(peer_rank, payload)
        except (ConnectionError, OSError):
            self._ready_event.set()

    def connect(self, peer_rank: int) -> P2PConnection:
        if peer_rank not in self._connections:
            self._connections[peer_rank] = P2PConnection(
                peer_rank, self._ctrl_sock, self._data_queue, self._ctrl_queue
            )
        return self._connections[peer_rank]

    def send_raw(self, peer_rank: int, data: bytes) -> None:
        conn = self.connect(peer_rank)
        conn.send_raw(data)

    def recv_ctrl(self, peer_rank: int, timeout: float = 30.0) -> bytes:
        return self._ctrl_queue.get(peer_rank, timeout=timeout)

    @property
    def rank(self) -> int:
        return self._rank

    def get_disconnected_peers(self) -> set[int]:
        return set(self._disconnected_peers)

    @classmethod
    def create_standalone(
        cls,
        cluster_name: str,
        world_size: int,
        is_coordinator: bool = False,
        uid: str = "",
        script_name: str = "",
    ) -> "P2PTransport":
        from ..swift.compile import ensure_compiled

        obj = cls.__new__(cls)
        obj._world_size = world_size
        obj._connections = {}
        obj._data_queue = RecvQueue()
        obj._ctrl_queue = RecvQueue()
        obj._disconnected_peers = set()
        obj._keepalive_running = False

        helper_path = ensure_compiled()
        ctrl_path = f"/tmp/grove_p2p_{os.getpid()}.sock"
        obj._ctrl_path = ctrl_path

        cmd = [str(helper_path), "mesh", cluster_name, str(world_size), ctrl_path]
        if is_coordinator:
            cmd += ["--coordinator", cluster_name, uid, script_name]

        log.info(f"Starting P2P helper: {' '.join(cmd[-4:])}")
        obj._process = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        obj._ctrl_sock = obj._connect_helper(ctrl_path)
        obj._ready_event = threading.Event()
        obj._ready_line = ""
        obj._reader_thread = threading.Thread(target=obj._read_loop, daemon=True)
        obj._reader_thread.start()

        ready_line = obj._wait_ready()
        parts = ready_line.split()
        if len(parts) != 2 or parts[0] != "ready":
            raise ValueError(f"Unexpected helper message: {ready_line!r}")
        obj._rank = int(parts[1])
        log.info(f"Rank {obj._rank}: P2P transport ready")
        return obj

    def start_keepalive(self, interval: float = 0.2) -> None:
        self._keepalive_running = True
        t = threading.Thread(
            target=self._keepalive_loop, args=(interval,), daemon=True,
        )
        t.start()

    def _keepalive_loop(self, interval: float) -> None:
        peer = (self._rank + 1) % self._world_size
        while self._keepalive_running:
            try:
                self.send_raw(peer, b"PING")
            except (ConnectionError, OSError):
                break
            time.sleep(interval)

    def close(self) -> None:
        self._keepalive_running = False
        self._connections.clear()
        try:
            self._ctrl_sock.close()
        except OSError:
            pass
        self._process.terminate()
        self._process.wait(timeout=5)
        if os.path.exists(self._ctrl_path):
            os.unlink(self._ctrl_path)

    @property
    def transport_type(self) -> str:
        return "p2p"
