"""AWDL discovery and control, with WiFi data plane upgrade when available."""

import os
import socket
import struct
import threading
import numpy as np
from .base import Transport, Connection
from .p2p import P2PTransport, P2PConnection
from .socket_conn import SocketConnection
from .._types import DEFAULT_BASE_PORT
from .._utils import get_logger, get_local_ip, configure_socket

log = get_logger("hybrid")

_WIFI_PORT_BASE = DEFAULT_BASE_PORT + 100
_PROBE_TIMEOUT = 2.0


class HybridConnection(Connection):
    def __init__(self, p2p_conn: P2PConnection) -> None:
        self._p2p = p2p_conn
        self._tcp: SocketConnection | None = None

    def send(self, buf: np.ndarray) -> None:
        if self._tcp is not None:
            try:
                self._tcp.send(buf)
                return
            except (ConnectionError, OSError):
                self._tcp = None
                raise
        self._p2p.send(buf)

    def recv(self, buf: np.ndarray) -> None:
        if self._tcp is not None:
            try:
                self._tcp.recv(buf)
                return
            except (ConnectionError, OSError):
                self._tcp = None
                raise
        self._p2p.recv(buf)

    def upgrade(self, tcp_conn: SocketConnection) -> None:
        self._tcp = tcp_conn

    def close(self) -> None:
        if self._tcp is not None:
            self._tcp.close()
        self._p2p.close()


class HybridTransport(Transport):
    def __init__(self, inner: P2PTransport) -> None:
        self._inner = inner
        self._connections: dict[int, HybridConnection] = {}
        self._tcp_sockets: dict[int, socket.socket] = {}
        self._upgraded = False

    @classmethod
    def create_standalone(
        cls,
        cluster_name: str,
        world_size: int,
        is_coordinator: bool = False,
        uid: str = "",
        script_name: str = "",
    ) -> "HybridTransport":
        inner = P2PTransport.create_standalone(
            cluster_name, world_size,
            is_coordinator=is_coordinator,
            uid=uid,
            script_name=script_name,
        )
        return cls(inner)

    def _try_wifi_upgrade(self) -> None:
        rank = self._inner.rank
        ws = self._inner._world_size

        lan_ip = get_local_ip()
        if lan_ip == "127.0.0.1":
            return

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        configure_socket(listener, tcp=True)
        try:
            listener.bind(("0.0.0.0", _WIFI_PORT_BASE + rank))
        except OSError:
            listener.bind(("0.0.0.0", 0))
        port = listener.getsockname()[1]
        listener.listen(ws)
        listener.settimeout(_PROBE_TIMEOUT + 1)

        my_addr = f"{lan_ip}:{port}".encode()
        for peer in range(ws):
            if peer != rank:
                self._inner.send_raw(peer, b"WIFI" + my_addr)

        peer_addrs: dict[int, tuple[str, int]] = {}
        try:
            for peer in range(ws):
                if peer != rank:
                    msg = self._inner.recv_ctrl(peer, timeout=5.0)
                    if msg[:4] != b"WIFI":
                        raise ValueError(f"Unexpected: {msg[:8]!r}")
                    host, port_str = msg[4:].decode().rsplit(":", 1)
                    peer_addrs[peer] = (host, int(port_str))
        except (TimeoutError, ValueError):
            listener.close()
            return

        success = True
        accepted: dict[int, socket.socket] = {}

        def accept_incoming():
            nonlocal success
            for _ in range(rank):
                try:
                    sock, _ = listener.accept()
                    configure_socket(sock, tcp=True)
                    data = sock.recv(4)
                    if len(data) == 4:
                        peer_rank = struct.unpack("<I", data)[0]
                        accepted[peer_rank] = sock
                    else:
                        success = False
                except (OSError, TimeoutError):
                    success = False

        accept_thread = None
        if rank > 0:
            accept_thread = threading.Thread(target=accept_incoming)
            accept_thread.start()

        for peer in range(rank + 1, ws):
            host, p = peer_addrs[peer]
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                configure_socket(sock, tcp=True)
                sock.settimeout(_PROBE_TIMEOUT)
                sock.connect((host, p))
                sock.sendall(struct.pack("<I", rank))
                sock.settimeout(120.0)
                self._tcp_sockets[peer] = sock
            except (OSError, TimeoutError):
                success = False
                break

        if accept_thread is not None:
            accept_thread.join(timeout=_PROBE_TIMEOUT + 2)

        self._tcp_sockets.update(accepted)
        listener.close()

        vote = b"\x01" if success else b"\x00"
        for peer in range(ws):
            if peer != rank:
                self._inner.send_raw(peer, b"VOTE" + vote)

        all_ok = success
        try:
            for peer in range(ws):
                if peer != rank:
                    msg = self._inner.recv_ctrl(peer, timeout=5.0)
                    if msg[:4] != b"VOTE" or msg[4:5] != b"\x01":
                        all_ok = False
        except TimeoutError:
            all_ok = False

        if all_ok:
            self._upgraded = True
            log.info(f"WiFi data plane active ({len(self._tcp_sockets)} connections)")
        else:
            for sock in self._tcp_sockets.values():
                try:
                    sock.close()
                except OSError:
                    pass
            self._tcp_sockets.clear()
            log.info("AWDL data plane active")

    def start_keepalive(self) -> None:
        if self._upgraded or self._inner._world_size <= 1:
            return
        self._inner.start_keepalive()

    def connect(self, peer_rank: int) -> HybridConnection:
        if peer_rank not in self._connections:
            p2p_conn = self._inner.connect(peer_rank)
            hybrid = HybridConnection(p2p_conn)
            if self._upgraded and peer_rank in self._tcp_sockets:
                hybrid.upgrade(SocketConnection(self._tcp_sockets[peer_rank]))
            self._connections[peer_rank] = hybrid
        return self._connections[peer_rank]

    def send_raw(self, peer_rank: int, data: bytes) -> None:
        self._inner.send_raw(peer_rank, data)

    def recv_ctrl(self, peer_rank: int, timeout: float = 30.0) -> bytes:
        return self._inner.recv_ctrl(peer_rank, timeout)

    @property
    def rank(self) -> int:
        return self._inner.rank

    @property
    def _process(self):
        return getattr(self._inner, '_process', None)

    def get_disconnected_peers(self) -> set[int]:
        return self._inner.get_disconnected_peers()

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        for sock in self._tcp_sockets.values():
            try:
                sock.close()
            except OSError:
                pass
        self._tcp_sockets.clear()
        self._inner.close()

    @property
    def transport_type(self) -> str:
        return "wifi" if self._upgraded else "awdl"
