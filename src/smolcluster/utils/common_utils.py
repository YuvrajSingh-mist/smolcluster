import pickle
import socket
import struct
import time
import logging
from typing import Any, Optional
from smolcluster.utils.layers import get_model_per_node    
import torch

# Module logger
logger = logging.getLogger(__name__)


class InferenceMetrics:
    """Track inference performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics for a new inference request."""
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.num_tokens = 0

    def start_inference(self):
        """Mark the start of inference."""
        self.start_time = time.time()

    def record_first_token(self):
        """Record when the first token was generated."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def record_token(self):
        """Record that a token was generated."""
        self.num_tokens += 1
        if self.num_tokens == 1:
            self.record_first_token()

    def end_inference(self):
        """Mark the end of inference."""
        self.end_time = time.time()

    def get_metrics(self) -> dict:
        """Calculate and return inference metrics."""
        metrics = {}

        if self.start_time and self.end_time:
            # Total time for generation
            total_time = self.end_time - self.start_time
            metrics["total_time_ms"] = round(total_time * 1000, 2)

            # Time to first token (TTFT)
            if self.first_token_time:
                ttft = self.first_token_time - self.start_time
                metrics["time_to_first_token_ms"] = round(ttft * 1000, 2)
            else:
                metrics["time_to_first_token_ms"] = 0

            # Tokens per second (throughput)
            if self.num_tokens > 0 and total_time > 0:
                metrics["tokens_per_second"] = round(self.num_tokens / total_time, 2)
            else:
                metrics["tokens_per_second"] = 0

            metrics["num_tokens"] = self.num_tokens

        return metrics


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
            metrics["send_bandwidth_mbps"] = (
                (total_send_mb * 8) / total_send_time if total_send_time > 0 else 0
            )
            metrics["avg_send_latency_ms"] = (
                sum(self.send_times) / len(self.send_times)
            ) * 1000
            metrics["total_send_mb"] = total_send_mb

        if self.recv_bytes:
            total_recv_mb = sum(self.recv_bytes) / (1024 * 1024)
            total_recv_time = sum(self.recv_times)
            metrics["recv_bandwidth_mbps"] = (
                (total_recv_mb * 8) / total_recv_time if total_recv_time > 0 else 0
            )
            metrics["avg_recv_latency_ms"] = (
                sum(self.recv_times) / len(self.recv_times)
            ) * 1000
            metrics["total_recv_mb"] = total_recv_mb

        if self.buffer_sizes:
            metrics["avg_buffer_size_kb"] = (
                sum(self.buffer_sizes) / len(self.buffer_sizes)
            ) / 1024
            metrics["max_buffer_size_kb"] = max(self.buffer_sizes) / 1024

        if reset:
            self.send_times.clear()
            self.recv_times.clear()
            self.send_bytes.clear()
            self.recv_bytes.clear()
            self.buffer_sizes.clear()
            self.last_log_time = time.time()

        return metrics


# Global metrics instances
_network_metrics = NetworkMetrics()
_inference_metrics = InferenceMetrics()


def get_network_metrics(reset: bool = True) -> dict:
    """Get current network metrics."""
    return _network_metrics.get_metrics(reset=reset)


def get_inference_metrics() -> InferenceMetrics:
    """Get the global inference metrics instance."""
    return _inference_metrics


def calculate_bandwidth_metrics(
    sizes: list[float],
    times: list[float],
    window_size: int
) -> dict[str, float]:
    """
    Calculate bandwidth metrics from transfer size and time lists.
    
    Args:
        sizes: List of transfer sizes in MB
        times: List of transfer times in seconds
        window_size: Number of recent samples to consider
        
    Returns:
        Dictionary with bandwidth_mbps and avg_size_mb
    """
    recent_sizes = sizes[-window_size:]
    recent_times = times[-window_size:]
    
    total_mb = sum(recent_sizes)
    total_time = sum(recent_times)
    
    bandwidth_mbps = (total_mb * 8) / total_time if total_time > 0 else 0
    avg_size_mb = total_mb / len(recent_sizes) if len(recent_sizes) > 0 else 0
    
    return {
        "bandwidth_mbps": bandwidth_mbps,
        "avg_size_mb": avg_size_mb
    }


def recv_tensor(sock):
    """Receive a tensor with network metrics tracking."""
    start_time = time.time()

    # read seq_len
    raw = sock.recv(4)
    if not raw:
        raise ConnectionError("socket closed")
    seq_len = struct.unpack(">I", raw)[0]

    # read payload length
    raw = sock.recv(4)
    payload_len = struct.unpack(">I", raw)[0]

    _network_metrics.record_buffer_size(payload_len)

    # read payload
    data = b""
    while len(data) < payload_len:
        chunk = sock.recv(min(4096, payload_len - len(data)))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk

    tensor = torch.frombuffer(data, dtype=torch.float32).view(1, seq_len, 768)

    # Record metrics
    duration = time.time() - start_time
    _network_metrics.record_recv(payload_len, duration)

    return tensor


def send_tensor(sock, tensor: torch.Tensor):
    """Send a tensor with network metrics tracking."""
    start_time = time.time()

    seq_len = tensor.shape[1]

    payload = tensor.detach().cpu().numpy().astype("float32").tobytes()

    _network_metrics.record_buffer_size(len(payload))

    sock.sendall(struct.pack(">I", seq_len))  # seq_len
    sock.sendall(struct.pack(">I", len(payload)))  # payload length
    sock.sendall(payload)

    # Record metrics
    duration = time.time() - start_time
    _network_metrics.record_send(len(payload), duration)


def send_message(
    sock: socket.SocketType, message: Any, buffer_size_mb: Optional[int] = None
) -> None:
    """Send a message with optional buffer size configuration and metrics tracking.

    Args:
        sock: Socket to send on
        message: Message to send (will be pickled)
        buffer_size_mb: Buffer size in MB (None = use 4MB default)
    """
    start_time = time.time()

    # Set buffer size (device-specific or default)
    buffer_bytes = (
        (buffer_size_mb * 1024 * 1024) if buffer_size_mb else (4 * 1024 * 1024)
    )
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


def receive_message(
    sock: socket.SocketType, buffer_size_mb: Optional[int] = None
) -> Optional[dict]:
    """Receive a message with optional buffer size configuration and metrics tracking.

    Args:
        sock: Socket to receive from
        buffer_size_mb: Buffer size in MB (None = use 4MB default)

    Returns:
        Unpickled message or None if socket closed
    """
    start_time = time.time()

    # Set buffer size (device-specific or default)
    buffer_bytes = (
        (buffer_size_mb * 1024 * 1024) if buffer_size_mb else (4 * 1024 * 1024)
    )
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


def set_weights_by_layer(
    weights_received_dict: dict[int, dict[str, torch.Tensor]],
    model: torch.nn.Module,
    worker_rank: int,
) -> None:
    """
    Update model with received weights from other workers (ZeRO Stage 1).
    Each worker sends only their owned parameters, so we just copy them into the model.
    
    Args:
        weights_received_dict: Dict of {rank: owned_state_dict} received from other workers
        model: Full model to update
       )
    """
    if not weights_received_dict:
        return
    
    # Build model parameter dict for fast lookup
    model_params = {name: param for name, param in model.named_parameters()}
    
    with torch.no_grad():
        for rank, state_dict in weights_received_dict.items():
            
            if rank == worker_rank:
                continue
            # Each rank sends only their owned parameters
            # Just copy all parameters from their state_dict into the model
            

            # logger.info(state_dict)
            for param_name, param_value in state_dict.items():
                
                if param_name.startswith('model.'):
                    param_name = param_name[len("model."):]
                    if param_name in list(model_params.keys()):
                        
                        model_params[param_name].data.copy_(param_value.to(model_params[param_name].device))
        