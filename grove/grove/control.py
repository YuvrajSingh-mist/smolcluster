"""Control plane wire protocol for coordinator-worker communication."""

import json
import socket
import struct
from enum import IntEnum
from ._utils import recvall

_CTRL_HEADER_FMT = "<HI"
_CTRL_HEADER_SIZE = struct.calcsize(_CTRL_HEADER_FMT)


class MsgType(IntEnum):
    HEARTBEAT = 0x01
    HEARTBEAT_ACK = 0x02
    MEMBERSHIP_UPDATE = 0x03
    REFORM_ACK = 0x04
    REFORM_COMPLETE = 0x05
    JOIN_REQUEST = 0x06
    JOIN_ACCEPTED = 0x07
    JOIN_REJECTED = 0x08
    STATS_UPDATE = 0x09
    SCRIPT_REQUEST = 0x0A
    SCRIPT_RESPONSE = 0x0B


CTRL_MAGIC = b"GCTL"


def encode_ctrl(msg_type: MsgType, payload: dict) -> bytes:
    data = json.dumps(payload).encode()
    return CTRL_MAGIC + struct.pack(_CTRL_HEADER_FMT, msg_type, len(data)) + data


def decode_ctrl(raw: bytes) -> tuple[MsgType, dict]:
    if raw[:4] != CTRL_MAGIC:
        raise ValueError(f"Bad ctrl magic: {raw[:4]!r}")
    msg_type, length = struct.unpack(_CTRL_HEADER_FMT, raw[4:4 + _CTRL_HEADER_SIZE])
    data = raw[4 + _CTRL_HEADER_SIZE:4 + _CTRL_HEADER_SIZE + length]
    return MsgType(msg_type), json.loads(data) if data else {}


def send_msg(sock: socket.socket, msg_type: MsgType, payload: dict) -> None:
    data = json.dumps(payload).encode()
    header = struct.pack(_CTRL_HEADER_FMT, msg_type, len(data))
    sock.sendall(header + data)


def recv_msg(sock: socket.socket) -> tuple[MsgType, dict]:
    header = recvall(sock, _CTRL_HEADER_SIZE)
    msg_type, length = struct.unpack(_CTRL_HEADER_FMT, header)
    data = recvall(sock, length) if length > 0 else b"{}"
    return MsgType(msg_type), json.loads(data)
