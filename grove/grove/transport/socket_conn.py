"""Socket-based connection with length-prefixed framing."""

import socket
import struct
import threading
import numpy as np
from .base import Connection
from .._types import MAGIC

_HEADER_FMT = "<4sIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _recv_into_buffer(sock: socket.socket, buf: memoryview) -> None:
    pos = 0
    total = len(buf)
    while pos < total:
        n = sock.recv_into(buf[pos:])
        if n == 0:
            raise ConnectionError("Connection closed during recv")
        pos += n


def _recv_bytes(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    _recv_into_buffer(sock, memoryview(buf))
    return bytes(buf)


class SocketConnection(Connection):
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._sock.settimeout(120.0)
        self._send_lock = threading.Lock()

    def send(self, buf: np.ndarray) -> None:
        flat = buf.ravel()
        shape_bytes = struct.pack(f"<{buf.ndim}q", *buf.shape)
        header = struct.pack(_HEADER_FMT, MAGIC, flat.nbytes, buf.dtype.num, buf.ndim)
        with self._send_lock:
            self._sock.sendall(header + shape_bytes)
            self._sock.sendall(memoryview(flat))

    def recv(self, buf: np.ndarray) -> None:
        header = _recv_bytes(self._sock, _HEADER_SIZE)
        magic, length, _dtype_num, ndim = struct.unpack(_HEADER_FMT, header)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        _recv_bytes(self._sock, ndim * 8)
        if buf.flags["C_CONTIGUOUS"]:
            _recv_into_buffer(self._sock, memoryview(buf).cast("B")[:length])
        else:
            tmp = np.empty(buf.shape, dtype=buf.dtype)
            _recv_into_buffer(self._sock, memoryview(tmp).cast("B")[:length])
            np.copyto(buf, tmp)

    @property
    def is_blocking(self) -> bool:
        return True

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
