"""smolcluster utilities — re-exports socket messaging helpers, gradient/weight accessors, dataset splitting, device selection, model sharding layers, model download utilities, logging, checkpointing, decoding, quantization, and CLI helpers."""
from .checkpointing import CheckpointManager, should_save_checkpoint
from .cli import (
    ALGORITHMS,
    MODES,
    build_discover_parser,
    build_main_parser,
    grove_world_size,
    parse_server_worker_mode,
    run_dashboard,
    should_autodiscover,
)
from .common_utils import (
    InferenceMetrics,
    NetworkMetrics,
    avg_grads,
    calculate_bandwidth_metrics,
    clear_skeleton_gradients,
    extract_owned_gradients,
    forward_through_shard,
    get_effective_decoding_strategies,
    get_gradients,
    get_inference_metrics,
    get_network_metrics,
    get_ordered_shard_layer_names,
    get_weights,
    load_model_and_tokenizer,
    load_params_into_skeleton,
    receive_message,
    recv_tensor,
    resolve_generation_request_params,
    send_message,
    send_tensor,
    set_gradients,
    set_weights,
    set_weights_by_layer,
    unload_params_from_skeleton,
)
from .data import get_data_indices
from .decoding import sample_next_token
from .device import get_device
from .layers import (
    get_expert_per_node,
    get_hfmodel_per_node,
    get_model_per_node,
    load_weights_per_node,
)
from .logging_utils import (
    emit_smol_event,
    emit_transport_event,
    setup_cluster_logging,
    setup_logging,
)
from .model_downloader import download_and_convert_model, ensure_model_weights
from .quantization import (
    calculate_compression_ratio,
    dequantize_model_weights,
    quantize_model_weights,
)

__all__ = [
    # checkpointing
    "CheckpointManager",
    "should_save_checkpoint",
    # cli
    "ALGORITHMS",
    "MODES",
    "build_discover_parser",
    "build_main_parser",
    "grove_world_size",
    "parse_server_worker_mode",
    "run_dashboard",
    "should_autodiscover",
    # common_utils
    "InferenceMetrics",
    "NetworkMetrics",
    "avg_grads",
    "calculate_bandwidth_metrics",
    "clear_skeleton_gradients",
    "extract_owned_gradients",
    "forward_through_shard",
    "get_effective_decoding_strategies",
    "get_gradients",
    "get_inference_metrics",
    "get_network_metrics",
    "get_ordered_shard_layer_names",
    "get_weights",
    "load_model_and_tokenizer",
    "load_params_into_skeleton",
    "receive_message",
    "recv_tensor",
    "resolve_generation_request_params",
    "send_message",
    "send_tensor",
    "set_gradients",
    "set_weights",
    "set_weights_by_layer",
    "unload_params_from_skeleton",
    # data
    "get_data_indices",
    # decoding
    "sample_next_token",
    # device
    "get_device",
    # layers
    "get_expert_per_node",
    "get_hfmodel_per_node",
    "get_model_per_node",
    "load_weights_per_node",
    # logging_utils
    "emit_smol_event",
    "emit_transport_event",
    "setup_cluster_logging",
    "setup_logging",
    # model_downloader
    "download_and_convert_model",
    "ensure_model_weights",
    # quantization
    "calculate_compression_ratio",
    "dequantize_model_weights",
    "quantize_model_weights",
]
