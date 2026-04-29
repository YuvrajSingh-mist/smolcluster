"""TCP transport for WiFi LAN communication."""

import socket
import struct
import threading
from .base import Transport
from .socket_conn import SocketConnection
from .._types import DEFAULT_BASE_PORT
from .._utils import get_logger, get_local_ip, recvall, configure_socket

log = get_logger("tcp")


class TCPTransport(Transport):
    def __init__(
        self,
        rank: int,
        world_size: int,
        store: "Store",
        host: str = "0.0.0.0",
        port: int = DEFAULT_BASE_PORT,
    ):
        self._rank = rank
        self._world_size = world_size
        self._store = store
        self._connections: dict[int, SocketConnection] = {}
        self._sockets: dict[int, socket.socket] = {}
        self._lock = threading.Lock()
        self._running = True

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        configure_socket(self._listener, tcp=True)
        self._listener.bind((host, port + rank))
        actual_port = self._listener.getsockname()[1]
        self._listener.listen(world_size + 8)

        addr = f"{get_local_ip()}:{actual_port}"
        store.set(f"tcp_addr/{rank}", addr.encode())
        store.wait([f"tcp_addr/{i}" for i in range(world_size)])
        self._establish_connections()

        self._listener_thread = threading.Thread(target=self._keep_listening, daemon=True)
        self._listener_thread.start()
        log.info(f"Rank {rank}: TCP transport ready at {addr}")

    def _establish_connections(self) -> None:
        accept_count = self._rank

        def accept_loop() -> None:
            for _ in range(accept_count):
                sock, _ = self._listener.accept()
                configure_socket(sock, tcp=True)
                rank_bytes = recvall(sock, 4)
                peer_rank = struct.unpack("<I", rank_bytes)[0]
                self._sockets[peer_rank] = sock

        if accept_count > 0:
            t = threading.Thread(target=accept_loop)
            t.start()

        for peer in range(self._rank + 1, self._world_size):
            addr = self._store.get(f"tcp_addr/{peer}").decode()
            host, port_str = addr.rsplit(":", 1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            configure_socket(sock, tcp=True)
            sock.connect((host, int(port_str)))
            sock.sendall(struct.pack("<I", self._rank))
            self._sockets[peer] = sock

        if accept_count > 0:
            t.join()

    def _keep_listening(self) -> None:
        while self._running:
            try:
                self._listener.settimeout(1.0)
                sock, _ = self._listener.accept()
                configure_socket(sock, tcp=True)
                rank_bytes = recvall(sock, 4)
                peer_rank = struct.unpack("<I", rank_bytes)[0]
                with self._lock:
                    self._sockets[peer_rank] = sock
                    self._connections.pop(peer_rank, None)
                log.info(f"Rank {self._rank}: accepted late connection from rank {peer_rank}")
            except socket.timeout:
                continue
            except OSError:
                break

    def connect(self, peer_rank: int) -> SocketConnection:
        with self._lock:
            if peer_rank not in self._connections:
                if peer_rank in self._sockets:
                    self._connections[peer_rank] = SocketConnection(self._sockets[peer_rank])
                else:
                    sock = self._connect_on_demand(peer_rank)
                    self._sockets[peer_rank] = sock
                    self._connections[peer_rank] = SocketConnection(sock)
            return self._connections[peer_rank]

    def _connect_on_demand(self, peer_rank: int) -> socket.socket:
        addr = self._store.get(f"tcp_addr/{peer_rank}").decode()
        host, port_str = addr.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_socket(sock, tcp=True)
        sock.connect((host, int(port_str)))
        sock.sendall(struct.pack("<I", self._rank))
        log.info(f"Rank {self._rank}: on-demand connect to rank {peer_rank}")
        return sock

    def disconnect(self, peer_rank: int) -> None:
        with self._lock:
            conn = self._connections.pop(peer_rank, None)
            if conn:
                conn.close()
            sock = self._sockets.pop(peer_rank, None)
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._running = False
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        self._sockets.clear()
        self._listener.close()

    @property
    def transport_type(self) -> str:
        return "tcp"
