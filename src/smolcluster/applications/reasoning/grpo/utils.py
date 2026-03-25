"""Generic MLX training utilities for GRPO — gradient helpers, tokenisation, data batching."""

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import re

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load as mlx_load

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metal memory logging
# ---------------------------------------------------------------------------

def _log_mem(tag: str) -> None:
    get_active = getattr(mx, "get_active_memory", mx.metal.get_active_memory)
    get_peak = getattr(mx, "get_peak_memory", mx.metal.get_peak_memory)
    logger.info(
        "[MEM] %s — active: %.0f MB  peak: %.0f MB",
        tag,
        get_active() / 1e6,
        get_peak() / 1e6,
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_dtype_from_config(config: Dict[str, Any]) -> type:
    """Map config dtype string ('float32', 'bfloat16') to an MLX dtype."""
    dtype_str = str(config.get("dtype", "float32")).lower()
    if dtype_str == "bfloat16":
        return mx.bfloat16
    if dtype_str == "float32":
        return mx.float32
    logger.warning("Unknown dtype '%s', defaulting to float32", dtype_str)
    return mx.float32


def get_mlx_device(config: Dict[str, Any]) -> mx.Device:
    """Return the MLX Device for config['device'] ('cpu', 'gpu', or 'metal')."""
    device_str = str(config.get("device", "cpu")).lower()
    return mx.gpu if device_str in ("gpu", "metal") else mx.cpu


# ---------------------------------------------------------------------------
# Data batching
# ---------------------------------------------------------------------------

def iterate_batches(
    examples: Sequence[Tuple[str, str]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[Tuple[List[str], List[str]]]:
    indices = np.arange(len(examples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if len(batch_indices) == 0:
            continue
        batch = [examples[idx] for idx in batch_indices]
        yield [q for q, _ in batch], [a for _, a in batch]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_rollouts(
    tokenizer: Any,
    rollout_texts: List[str],
    max_length: int,
    device: mx.Device = mx.cpu,
) -> Tuple[mx.array, mx.array]:
    encoded = [list(tokenizer.encode(text))[:max_length] for text in rollout_texts]
    if not encoded:
        with mx.stream(mx.default_stream(device)):
            return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    target_length = max(len(t) for t in encoded)

    input_rows, mask_rows = [], []
    for tokens in encoded:
        pad_len = target_length - len(tokens)
        input_rows.append(tokens + [pad_token_id] * pad_len)
        mask_rows.append([1] * len(tokens) + [0] * pad_len)

    with mx.stream(mx.default_stream(device)):
        return mx.array(input_rows, dtype=mx.int32), mx.array(mask_rows, dtype=mx.int32)


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


def _scale_grads(grads: Any, scale: float) -> Any:
    return tree_unflatten([(k, v * scale) for k, v in tree_flatten(grads)])


def _add_grads(acc: Any, new: Any) -> Any:
    new_flat = dict(tree_flatten(new))
    return tree_unflatten([(k, v + new_flat[k]) for k, v in tree_flatten(acc)])

# ---------------------------------------------------------------------------
# Rewards Parsing
# ---------------------------------------------------------------------------

def parse_numeric_answer(text: str) -> float:
    matches = re.findall(r"<answer>\s*([-+]?\d*\.?\d+)\s*<\/answer>", text, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1])

    final_answer_match = re.findall(
        r"final\s+answer[^\d\-\+]*([-+]?\d*\.?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return float(final_answer_match[-1])

    fallback_numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if fallback_numbers:
        return float(fallback_numbers[-1])

    return float("nan")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    dtype: type,
    config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Tuple[Any, Optional[Any], Any]:
    """Load the policy model (and optionally the reference model) from HF.

    Args:
        dtype:        MLX dtype for the model (e.g. mx.bfloat16).
        config:       GRPO training config dict.
        model_config: Model config dict (contains ``dp.hf_model_name``).

    Returns:
        (model, ref_model, tokenizer) — ref_model is None when use_kl=false.
    """
    model_name = model_config["dp"]["hf_model_name"]
    tokenizer_config = {
        "trust_remote_code": True if config.get("trust_remote_code", False) else None
    }

    device = get_mlx_device(config)
    device_stream = mx.default_stream(device)
    logger.info("Loading MLX model: %s (device=%s)", model_name, config.get("device", "cpu"))

    model, tokenizer = mlx_load(model_name, tokenizer_config=tokenizer_config)
    with mx.stream(device_stream):
        mx.eval(model.parameters())
    _log_mem("load_model: after policy model load")

    ref_model: Optional[Any] = None
    if config.get("use_kl", True):
        logger.info("Loading reference model (use_kl=true) ...")
        ref_model, _ = mlx_load(model_name, tokenizer_config=tokenizer_config)
        ref_model.eval()
        with mx.stream(device_stream):
            mx.eval(ref_model.parameters())
    else:
        logger.info("Skipping reference model load (use_kl=false)")

    return model, ref_model, tokenizer
