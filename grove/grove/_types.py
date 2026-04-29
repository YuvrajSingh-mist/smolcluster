"""Core types and constants."""

from enum import Enum


class ReduceOp(Enum):
    SUM = "sum"
    PROD = "prod"
    MIN = "min"
    MAX = "max"


class TransportType(Enum):
    TCP = "tcp"
    P2P = "p2p"


MAGIC = b"GROV"
DEFAULT_SOCK_BUF_SIZE = 32 * 1024 * 1024
DEFAULT_BASE_PORT = 29500
