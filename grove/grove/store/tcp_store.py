"""TCP key-value store. Rank 0 runs server, all ranks connect as clients."""

import socket
import struct
import threading
import time
from enum import IntEnum
from .base import Store
from .._types import DEFAULT_BASE_PORT
from .._utils import get_logger, recvall, configure_socket

log = get_logger("tcp_store")


class _Op(IntEnum):
    SET = 1
    GET = 2
    WAIT = 3
    DELETE = 4
    CHECK = 5


class _Status(IntEnum):
    OK = 0
    NOT_FOUND = 1


class _TCPStoreServer:
    def __init__(self, host: str, port: int):
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._waiters: list[tuple[set[str], threading.Event]] = []
        self._running = True

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        configure_socket(self._sock, tcp=True)
        self._sock.bind((host, port))
        self._sock.listen(64)
        self._sock.settimeout(1.0)

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        log.info(f"TCPStore server listening on {host}:{port}")

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._sock.accept()
                t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            while self._running:
                op_byte = conn.recv(1)
                if not op_byte:
                    break
                op = op_byte[0]

                key_len = struct.unpack("<I", recvall(conn, 4))[0]
                key = recvall(conn, key_len).decode() if key_len > 0 else ""
                val_len = struct.unpack("<I", recvall(conn, 4))[0]
                val = recvall(conn, val_len) if val_len > 0 else b""

                if op == _Op.SET:
                    with self._lock:
                        self._data[key] = val
                        self._check_waiters()
                    conn.sendall(struct.pack("<BI", _Status.OK, 0))

                elif op == _Op.GET:
                    with self._lock:
                        v = self._data.get(key)
                    if v is not None:
                        conn.sendall(struct.pack("<BI", _Status.OK, len(v)) + v)
                    else:
                        conn.sendall(struct.pack("<BI", _Status.NOT_FOUND, 0))

                elif op == _Op.WAIT:
                    keys = set(val.decode().split(",")) if val else set()
                    event = threading.Event()
                    with self._lock:
                        missing = keys - self._data.keys()
                        if not missing:
                            event.set()
                        else:
                            self._waiters.append((keys, event))
                    event.wait(timeout=30.0)
                    status = _Status.OK if event.is_set() else _Status.NOT_FOUND
                    conn.sendall(struct.pack("<BI", status, 0))

                elif op == _Op.DELETE:
                    with self._lock:
                        self._data.pop(key, None)
                    conn.sendall(struct.pack("<BI", _Status.OK, 0))

                elif op == _Op.CHECK:
                    with self._lock:
                        exists = key in self._data
                    status = _Status.OK if exists else _Status.NOT_FOUND
                    conn.sendall(struct.pack("<BI", status, 0))

        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    def _check_waiters(self) -> None:
        remaining = []
        for keys, event in self._waiters:
            if keys.issubset(self._data.keys()):
                event.set()
            else:
                remaining.append((keys, event))
        self._waiters = remaining

    def close(self) -> None:
        self._running = False
        self._sock.close()
        self._thread.join(timeout=2.0)


class TCPStore(Store):
    def __init__(
        self,
        rank: int,
        world_size: int,
        host: str = "127.0.0.1",
        port: int = DEFAULT_BASE_PORT - 100,
        timeout: float = 30.0,
    ):
        self._rank = rank
        self._timeout = timeout
        self._server: _TCPStoreServer | None = None
        self._lock = threading.Lock()

        if rank == 0:
            self._server = _TCPStoreServer(host, port)

        self._host = host
        self._port = port
        self._sock = self._connect_to_server()

    def _connect_to_server(self) -> socket.socket:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                configure_socket(sock, tcp=True)
                sock.connect((self._host, self._port))
                return sock
            except ConnectionRefusedError:
                time.sleep(0.01)
        raise TimeoutError(f"Could not connect to TCPStore at {self._host}:{self._port}")

    def _request(self, op: int, key: str = "", val: bytes = b"") -> tuple[int, bytes]:
        key_bytes = key.encode()
        msg = struct.pack("<B", op)
        msg += struct.pack("<I", len(key_bytes)) + key_bytes
        msg += struct.pack("<I", len(val)) + val
        with self._lock:
            self._sock.sendall(msg)
            resp = recvall(self._sock, 5)
            status, val_len = struct.unpack("<BI", resp)
            resp_val = recvall(self._sock, val_len) if val_len > 0 else b""
        return status, resp_val

    def set(self, key: str, value: bytes) -> None:
        status, _ = self._request(_Op.SET, key, value)
        if status != _Status.OK:
            raise RuntimeError(f"TCPStore set failed for key '{key}'")

    def get(self, key: str) -> bytes:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            status, val = self._request(_Op.GET, key)
            if status == _Status.OK:
                return val
            time.sleep(0.001)
        raise TimeoutError(f"Key '{key}' not found within {self._timeout}s")

    def get_nowait(self, key: str) -> bytes:
        status, val = self._request(_Op.GET, key)
        if status == _Status.OK:
            return val
        raise KeyError(key)

    def wait(self, keys: list[str], timeout: float | None = None) -> None:
        keys_str = ",".join(keys)
        status, _ = self._request(_Op.WAIT, val=keys_str.encode())
        if status != _Status.OK:
            raise TimeoutError(f"Wait timed out for keys: {keys}")

    def delete(self, key: str) -> None:
        self._request(_Op.DELETE, key)

    def close(self) -> None:
        self._sock.close()
        if self._server:
            self._server.close()
