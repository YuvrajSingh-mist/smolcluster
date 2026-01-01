import socket
import pickle, struct
from typing import Dict, Any
import torch

def send_message(sock: socket.SocketType, message: Any) -> None:
    
    data = pickle.dumps(message)
    sock.sendall(struct.pack('>I', len(data)) + data)
    
        
def receive_message(sock: socket.SocketType) -> dict:
    
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = b""
    while True:
        chunk = sock.recv(msglen)
        # print(len(chunk))
        # print(msglen)
        data += chunk
        if msglen > 0:
            msglen -= len(chunk)
        else:
            break
    return pickle.loads(data)
    

def get_gradients(model : torch.nn.Module) -> Dict[str, torch.Tensor]:
    
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().clone()
    return grads     


def set_weights(grads: Dict[str, torch.Tensor], model : torch.nn.Module):
    
    for name, param in model.named_parameters():
        if name in grads:
            grads[name] = grads[name].to(param.device)
            param.grad = grads[name].clone()
    