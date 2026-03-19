import pickle
import socket
import struct
import time
import logging
from copy import deepcopy
from typing import Any, Optional
import torch
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
        
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


def load_model_and_tokenizer(
    hf_model_name: str,
    device: Any,
    hf_token: Optional[str] = None,
    tokenizer_cfg: Optional[dict[str, Any]] = None,
    load_model: bool = True,
    load_tokenizer: bool = True,
    logger: Optional[logging.Logger] = None,
) -> tuple[Optional[Any], Optional[Any]]:
    """Load a causal LM on target device, and optionally its tokenizer.

    Args:
        hf_model_name: HuggingFace model identifier.
        device: Target torch device.
        hf_token: Optional HuggingFace token.
        tokenizer_cfg: Optional kwargs for tokenizer loading.
        load_model: Whether to load model. If False, only tokenizer is loaded.
        load_tokenizer: Whether to also load tokenizer.
        logger: Optional logger for status messages.

    Returns:
        Tuple of (model or None, tokenizer or None).
    """
  
    model = None
    if load_model:
        if logger:
            logger.info(f"Loading model from {hf_model_name} on {device}")

        model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=hf_token)  # type: ignore[call-arg]
        model = model.to(device)
        model.eval()

    tokenizer = None
    if load_tokenizer:
        if logger:
            logger.info(f"Loading tokenizer from {hf_model_name}")

        use_hf_defaults, tokenizer_kwargs = parse_tokenizer_config(tokenizer_cfg)

        if logger and use_hf_defaults:
            logger.info("Tokenizer loading with HuggingFace defaults enabled")

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            token=hf_token,
            **tokenizer_kwargs,
        )

    return model, tokenizer


def parse_tokenizer_config(
    tokenizer_cfg: Optional[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Return tokenizer mode and kwargs from config.

    Supports both:
    - tokenizer.overrides.{...}
    - legacy flat tokenizer keys
    """
    cfg = tokenizer_cfg or {}
    use_hf_defaults = bool(cfg.get("use_hf_defaults", True))

    overrides = cfg.get("overrides")
    if isinstance(overrides, dict):
        return use_hf_defaults, dict(overrides)

    tokenizer_kwargs = {
        key: value
        for key, value in cfg.items()
        if key not in {"use_hf_defaults", "overrides", "decoding_overrides"}
    }
    return use_hf_defaults, tokenizer_kwargs


def get_generation_config_defaults(
    hf_model_name: str,
    hf_token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Extract generation config defaults from HuggingFace model.
    
    This function retrieves the model's generation_config which contains
    default parameters like top_p, top_k, temperature, etc. These are the
    model author's recommended generation parameters.
    
    Args:
        hf_model_name: HuggingFace model identifier.
        hf_token: Optional HuggingFace token.
        logger: Optional logger for status messages.
    
    Returns:
        Dictionary with keys like 'top_p', 'top_k', 'temperature', etc.
        Returns empty dict if generation_config not available.

    """
    try:
      
        gen_cfg = GenerationConfig.from_pretrained(hf_model_name, token=hf_token)
        
        # Extract relevant generation parameters
        result = {}
        for param in ['top_p', 'top_k', 'temperature', 'repetition_penalty', 'do_sample']:
            if hasattr(gen_cfg, param):
                value = getattr(gen_cfg, param)
                if value is not None:
                    result[param] = value
        
        if logger and result:
            logger.info(f"Loaded generation config from {hf_model_name}: {result}")
        
        return result
    except Exception as e:
        if logger:
            logger.warning(f"Could not load generation_config for {hf_model_name}: {e}")
        
        return {}


def get_effective_decoding_strategies(
    model_cfg: dict[str, Any],
    hf_token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, dict[str, Any]]:
    """Build effective decoding strategy defaults for inference.

    Rules:
    - If tokenizer.use_hf_defaults=true, base values come from HF generation_config.
    - decoding_strategies acts as overrides.
    - tokenizer.decoding_overrides can override globally or per strategy.
    """
    strategies = deepcopy(model_cfg.get("decoding_strategies", {}))
    if not isinstance(strategies, dict):
        strategies = {}

    for name in ("greedy", "sampling", "top_p", "top_k"):
        if name not in strategies or not isinstance(strategies[name], dict):
            strategies[name] = {}

    tokenizer_cfg = model_cfg.get("tokenizer", {}) or {}
    use_hf_defaults = bool(tokenizer_cfg.get("use_hf_defaults", True))

    hf_defaults: dict[str, Any] = {}
    if use_hf_defaults:
        hf_defaults = get_generation_config_defaults(
            model_cfg.get("hf_model_name", ""),
            hf_token=hf_token,
            logger=logger,
        )

    if use_hf_defaults and hf_defaults:
        hf_temp = hf_defaults.get("temperature")
        if hf_temp is not None:
            for name in ("greedy", "sampling", "top_p", "top_k"):
                strategies[name]["temperature"] = hf_temp

        hf_top_p = hf_defaults.get("top_p")
        if hf_top_p is not None:
            strategies["top_p"]["p"] = hf_top_p

        hf_top_k = hf_defaults.get("top_k")
        if hf_top_k is not None:
            strategies["top_k"]["k"] = hf_top_k

    decoding_overrides = tokenizer_cfg.get("decoding_overrides", {})
    if isinstance(decoding_overrides, dict):
        global_temp = decoding_overrides.get("temperature")
        if global_temp is not None:
            for name in ("greedy", "sampling", "top_p", "top_k"):
                strategies[name]["temperature"] = global_temp

        override_top_p = decoding_overrides.get("top_p")
        if override_top_p is not None:
            strategies["top_p"]["p"] = override_top_p

        override_top_k = decoding_overrides.get("top_k")
        if override_top_k is not None:
            strategies["top_k"]["k"] = override_top_k

        per_strategy = decoding_overrides.get("strategies", {})
        if isinstance(per_strategy, dict):
            for name, values in per_strategy.items():
                if isinstance(values, dict):
                    strategies.setdefault(name, {}).update(values)

    if logger:
        active_raw = model_cfg.get("active_decoding_strategy", "greedy")
        if isinstance(active_raw, str):
            logger.info(
                "Effective decoding defaults (active=%s): %s",
                active_raw,
                strategies.get(active_raw, {}),
            )
        else:
            logger.warning(
                "Invalid active_decoding_strategy type: %s (value=%r)",
                type(active_raw).__name__,
                active_raw,
            )

    return strategies


def resolve_generation_request_params(
    payload: dict[str, Any],
    model_cfg: dict[str, Any],
    effective_strategies: dict[str, dict[str, Any]],
) -> tuple[int, str, float, float, int]:
    """Resolve generation params from request with model/HF defaults fallback.

    Raises ValueError when required values are missing or invalid.
    """
    active = payload.get("decoding_strategy")
    if not isinstance(active, str) or not active:
        active = model_cfg.get("active_decoding_strategy")

    if not isinstance(active, str) or not active:
        raise ValueError(
            "Missing decoding_strategy (request decoding_strategy or model config active_decoding_strategy)."
        )

    active_cfg = effective_strategies.get(active)
    if not isinstance(active_cfg, dict):
        raise ValueError(f"Unknown decoding strategy '{active}'.")

    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        max_tokens = model_cfg.get("max_new_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer (request max_tokens or model config max_new_tokens).")

    temperature = payload.get("temperature")
    if temperature is None:
        temperature = active_cfg.get("temperature")
    if not isinstance(temperature, (int, float)):
        raise ValueError(f"Missing temperature for strategy '{active}'.")

    top_p = payload.get("top_p")
    if top_p is None:
        top_p = active_cfg.get("p")
    if active == "top_p" and not isinstance(top_p, (int, float)):
        raise ValueError("Missing top_p for top_p decoding strategy.")
    if top_p is None:
        top_p = 0.0

    top_k = payload.get("top_k")
    if top_k is None:
        top_k = active_cfg.get("k")
    if active == "top_k" and not isinstance(top_k, int):
        raise ValueError("Missing top_k for top_k decoding strategy.")
    if top_k is None:
        top_k = 0

    return int(max_tokens), active, float(temperature), float(top_p), int(top_k)


def get_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if model is None:
        return {}
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().clone()
    return grads


def set_gradients(grads: dict[str, torch.Tensor], model: torch.nn.Module):
    for name, param in model.named_parameters():
        if name in grads:
            grads[name] = grads[name].to(param.device)
            param.grad = grads[name].clone()


def avg_grads(incoming_grads: dict[str, torch.Tensor], model: torch.nn.Module, num_workers: int):
    """
    Incrementally accumulate gradients with averaging factor.
    Takes incoming gradients, divides by num_workers, and adds to current gradients.
    This allows accumulating gradients worker-by-worker without creating full gradients at once.
    
    Args:
        incoming_grads: Dictionary of gradients from a peer worker
        model: The model whose gradients will be accumulated
        num_workers: Total number of workers for averaging
    """
    for name, param in model.named_parameters():
        if name in incoming_grads and param.grad is not None:
            # Move incoming gradient to the same device as the parameter
            incoming_grad = incoming_grads[name].to(param.device)
            # Add the scaled incoming gradient to the current gradient
            param.grad += incoming_grad / num_workers


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
    Update model with received weights from other workers (ZeRO Stage 1 + 2).
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
            
            for param_name, param_value in state_dict.items():
                
                if param_name.startswith('model.'):
                    param_name = param_name[len("model."):]
                    if param_name in list(model_params.keys()):
                        
                        model_params[param_name].data.copy_(param_value.to(model_params[param_name].device))
                        
                else:
                    if param_name in list(model_params.keys()):
                        model_params[param_name].data.copy_(param_value.to(model_params[param_name].device))


# FSDP Stage 3 Helper Functions

def load_params_into_skeleton(model: Any, params: dict, device: torch.device) -> None:
    """Load parameter shard into model skeleton."""
    with torch.no_grad():
        for layer_name, param_data in params.items():
            # `params` contains raw tensors keyed by full parameter name, e.g.
            # `model.blocks.0.qkv_proj.weight`.
            # The skeleton already has the module structure on the target device,
            # but its `.data` tensors are empty placeholders. We walk the module
            # path, find the destination parameter object in the skeleton, and
            # replace only its underlying storage with the received shard tensor.
            # This lets us reuse the same model object for forward/backward without
            # instantiating a second full model per worker.
            clean_name = layer_name.replace('model.', '', 1)
            parts = clean_name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            
            # Materialize this shard tensor on the execution device and mark it as
            # trainable so autograd can attach gradients during backward.
            param = getattr(module, parts[-1])
            param.data = param_data.to(device)
            param.requires_grad_(True)


def unload_params_from_skeleton(model: Any) -> None:
    """Clear parameters from skeleton to free memory."""
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.empty(0, device='cpu')


def get_ordered_shard_layer_names(model: Any, rank: int, num_workers: int) -> list[str]:
    """Return the canonical execution order for a worker shard."""
    total_layers = len(model.blocks)
    split_indices = torch.chunk(torch.arange(total_layers), num_workers)

    shard_layers: list[str] = []

    if rank == 0:
        shard_layers.extend(["token_embedding", "position_embedding"])

    for layer_idx in split_indices[rank].tolist():
        shard_layers.append(f"blocks.{layer_idx}")

    if rank == num_workers - 1:
        shard_layers.extend(["ln_f", "lm_head"])

    return shard_layers


def forward_through_shard(
    model: Any,
    activations: torch.Tensor,
    rank: int,
    num_workers: int,
    device: torch.device
) -> torch.Tensor:
    """Forward through one worker shard using parameters already loaded into the skeleton."""
    shard_layer_names = get_ordered_shard_layer_names(model, rank, num_workers)
    out = activations.to(device)

    for layer_name in shard_layer_names:
        if layer_name == 'token_embedding':
            if out.dtype != torch.long:
                raise RuntimeError(
                    f"Worker {rank}: token_embedding expects integer token ids, got {out.dtype}. "
                    f"This usually means rank-to-layer ownership is wrong for this shard: {shard_layer_names}"
                )
            out = model.token_embedding(out)
        elif layer_name == 'position_embedding':
            pos_ids = torch.arange(out.shape[1], dtype=torch.long, device=device)
            out = out + model.position_embedding(pos_ids)
        elif layer_name.startswith('blocks.'):
            block_idx = int(layer_name.split('.')[1])
            output = model.blocks[block_idx](out)
            out = output[0] if isinstance(output, tuple) else output
        elif layer_name == 'ln_f':
            out = model.ln_f(out)
        elif layer_name == 'lm_head':
            out = model.lm_head(out)

    return out


def extract_owned_gradients(model_skeleton: Any, own_params: dict) -> dict[str, torch.Tensor]:
    """Extract gradients for owned parameters from model skeleton."""
    grads = {}
    for layer_name in own_params:
        clean_name = layer_name.replace('model.', '', 1) if layer_name.startswith('model.') else layer_name
        module = model_skeleton
        for part in clean_name.split('.')[:-1]:
            module = getattr(module, part)
        param_name = clean_name.split('.')[-1]
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if param.grad is not None:
                grads[layer_name] = param.grad.detach().clone()
    return grads


def clear_skeleton_gradients(model_skeleton: Any) -> None:
    """Clear all gradients from model skeleton."""
    for param in model_skeleton.parameters():
        if param.grad is not None:
            param.grad = None

