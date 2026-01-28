import pickle
import socket
import struct
import time
from typing import Any, Optional

import torch


class NetworkMetrics:
    """Track network performance metrics for distributed training."""
    
    def __init__(self):
        self.send_times = []
        self.recv_times = []
        self.send_bytes = []
        self.recv_bytes = []
        self.buffer_sizes = []
        self.last_log_time = time.time()
    
    def record_send(self, num_bytes: int, duration: float):
        """Record a send operation."""
        self.send_bytes.append(num_bytes)
        self.send_times.append(duration)
    
    def record_recv(self, num_bytes: int, duration: float):
        """Record a receive operation."""
        self.recv_bytes.append(num_bytes)
        self.recv_times.append(duration)
    
    def record_buffer_size(self, size: int):
        """Record current buffer size."""
        self.buffer_sizes.append(size)
    
    def get_metrics(self, reset: bool = True) -> dict:
        """Get aggregated metrics and optionally reset counters."""
        metrics = {}
        
        if self.send_bytes:
            total_send_mb = sum(self.send_bytes) / (1024 * 1024)
            total_send_time = sum(self.send_times)
            metrics['send_bandwidth_mbps'] = (total_send_mb * 8) / total_send_time if total_send_time > 0 else 0
            metrics['avg_send_latency_ms'] = (sum(self.send_times) / len(self.send_times)) * 1000
            metrics['total_send_mb'] = total_send_mb
        
        if self.recv_bytes:
            total_recv_mb = sum(self.recv_bytes) / (1024 * 1024)
            total_recv_time = sum(self.recv_times)
            metrics['recv_bandwidth_mbps'] = (total_recv_mb * 8) / total_recv_time if total_recv_time > 0 else 0
            metrics['avg_recv_latency_ms'] = (sum(self.recv_times) / len(self.recv_times)) * 1000
            metrics['total_recv_mb'] = total_recv_mb
        
        if self.buffer_sizes:
            metrics['avg_buffer_size_kb'] = (sum(self.buffer_sizes) / len(self.buffer_sizes)) / 1024
            metrics['max_buffer_size_kb'] = max(self.buffer_sizes) / 1024
        
        if reset:
            self.send_times.clear()
            self.recv_times.clear()
            self.send_bytes.clear()
            self.recv_bytes.clear()
            self.buffer_sizes.clear()
            self.last_log_time = time.time()
        
        return metrics


# Global metrics instance (can be replaced with per-socket tracking if needed)
_network_metrics = NetworkMetrics()


def get_network_metrics(reset: bool = True) -> dict:
    """Get current network metrics."""
    return _network_metrics.get_metrics(reset=reset)


def send_message(sock: socket.SocketType, message: Any, buffer_size_mb: Optional[int] = None) -> None:
    """Send a message with optional buffer size configuration and metrics tracking.
    
    Args:
        sock: Socket to send on
        message: Message to send (will be pickled)
        buffer_size_mb: Buffer size in MB (None = use 4MB default)
    """
    start_time = time.time()
    
    # Set buffer size (device-specific or default)
    buffer_bytes = (buffer_size_mb * 1024 * 1024) if buffer_size_mb else (4 * 1024 * 1024)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
    except OSError:
        pass  # Use system default if unable to set
    
    data = pickle.dumps(message)
    _network_metrics.record_buffer_size(len(data))
    sock.sendall(struct.pack(">I", len(data)) + data)
    
    # Record metrics
    duration = time.time() - start_time
    _network_metrics.record_send(len(data), duration)


def receive_message(sock: socket.SocketType, buffer_size_mb: Optional[int] = None) -> dict:
    """Receive a message with optional buffer size configuration and metrics tracking.
    
    Args:
        sock: Socket to receive from
        buffer_size_mb: Buffer size in MB (None = use 4MB default)
        
    Returns:
        Unpickled message
    """
    start_time = time.time()
    
    # Set buffer size (device-specific or default)
    buffer_bytes = (buffer_size_mb * 1024 * 1024) if buffer_size_mb else (4 * 1024 * 1024)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
    except OSError:
        pass  # Use system default if unable to set
    
    # Read the 4-byte message length header
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None

    msglen = struct.unpack(">I", raw_msglen)[0]
    _network_metrics.record_buffer_size(msglen)

    # Read the message data - use smaller chunks for better cross-platform compatibility
    # Chunk size based on buffer size: 1MB for small buffers, up to 4MB for large buffers
    chunk_size_base = min(buffer_bytes // 4, 4 * 1024 * 1024)
    
    data = b""
    remaining = msglen
    while remaining > 0:
        chunk_size = min(chunk_size_base, remaining)
        chunk = sock.recv(chunk_size)
        if not chunk:
            raise ConnectionError("Socket connection broken while receiving message")
        data += chunk
        remaining -= len(chunk)

    result = pickle.loads(data)
    
    # Record metrics
    duration = time.time() - start_time
    _network_metrics.record_recv(msglen, duration)
    
    return result


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
