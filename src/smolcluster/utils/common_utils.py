import pickle
import socket
import struct
from typing import Any

import torch


def send_message(sock: socket.SocketType, message: Any) -> None:
    # Optimize socket for high-bandwidth transfers (40Gbps Thunderbolt)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)  # 4MB send buffer (cross-platform)
    except OSError:
        pass  # Use system default if unable to set
    
    data = pickle.dumps(message)
    sock.sendall(struct.pack(">I", len(data)) + data)


def receive_message(sock: socket.SocketType) -> dict:
    # Optimize socket for high-bandwidth transfers (40Gbps Thunderbolt)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)  # 4MB receive buffer (cross-platform)
    except OSError:
        pass  # Use system default if unable to set
    
    # Read the 4-byte message length header
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None

    msglen = struct.unpack(">I", raw_msglen)[0]

    # Read the message data in larger chunks for 40Gbps connections
    data = b""
    remaining = msglen
    while remaining > 0:
        chunk_size = min(1 * 1024 * 1024, remaining)  # 1MB chunks (works reliably on Windows/macOS/Linux)
        chunk = sock.recv(chunk_size)
        if not chunk:
            raise ConnectionError("Socket connection broken while receiving message")
        data += chunk
        remaining -= len(chunk)

    return pickle.loads(data)


def get_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().clone()
    return grads


def set_gradients(grads: dict[str, torch.Tensor], model: torch.nn.Module):
    for name, param in model.named_parameters():
        if name in grads:
            if param.grad is not None:
                grads[name] = grads[name].to(param.device)
                param.grad = grads[name].clone()



def set_weights(
    weights: dict[str, torch.Tensor], model: torch.nn.Module, grad_scaling: float = 0.0
) -> torch.nn.Module:
    curr_weights = get_weights(model)
    for name, param in model.named_parameters():
        if name in weights:
            weights[name] = weights[name].to(param.device)
            if grad_scaling != 0.0:
                param.data = grad_scaling * curr_weights[name].clone() + (
                    1 - grad_scaling
                ) * weights[name].to(param.device)
            else:
                param.data = weights[name].clone()

    return model


def get_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = torch.Tensor(param.data.detach().cpu().clone())
    return weights
