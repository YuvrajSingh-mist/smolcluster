import pickle
import socket
import struct
from typing import Any

import torch


def send_message(sock: socket.SocketType, message: Any) -> None:
    data = pickle.dumps(message)
    sock.sendall(struct.pack(">I", len(data)) + data)


def receive_message(sock: socket.SocketType) -> dict:
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None

    msglen = struct.unpack(">I", raw_msglen)[0]
    data = b""
    while True:
        chunk = sock.recv(msglen)

        data += chunk
        if msglen > 0:
            msglen -= len(chunk)
        else:
            break
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


def set_weights(weights: dict[str, torch.Tensor], model: torch.nn.Module, grad_scaling: float=0.0) -> torch.nn.Module:
    
    curr_weights = get_weights(model)
    for name, param in model.named_parameters():
        if name in weights:
            weights[name] = weights[name].to(param.device)
            if grad_scaling != 0.0:
                param.data = grad_scaling * curr_weights[name].clone() + (1 - grad_scaling) * weights[name].to(param.device)
            else:
                param.data = weights[name].clone()
                
    return model

def get_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.detach().cpu().clone()
    return weights

