"""Shared utilities."""

import logging
import socket
from ._types import DEFAULT_SOCK_BUF_SIZE

_log = logging.getLogger("grove.utils")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"grove.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[grove %(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        nbytes = sock.recv_into(view[pos:])
        if nbytes == 0:
            raise ConnectionError("Connection closed during recv")
        pos += nbytes
    return bytes(buf)


def configure_socket(sock: socket.socket, *, tcp: bool = False) -> None:
    if tcp:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, 0x10, 5)   # TCP_KEEPIDLE: 5s
            sock.setsockopt(socket.IPPROTO_TCP, 0x101, 2)  # TCP_KEEPINTVL: 2s
            sock.setsockopt(socket.IPPROTO_TCP, 0x102, 3)  # TCP_KEEPCNT: 3 probes
        except OSError as e:
            _log.debug("Failed to set TCP keepalive: %s", e)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, DEFAULT_SOCK_BUF_SIZE)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, DEFAULT_SOCK_BUF_SIZE)
    except OSError as e:
        _log.debug("Failed to set socket buffer size: %s", e)
