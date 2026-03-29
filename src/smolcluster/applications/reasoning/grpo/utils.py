"""Generic MLX training utilities for GRPO — gradient helpers, tokenisation, data batching."""

import logging
from pathlib import Path
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

def _unwrap_tokenizer(tokenizer: Any) -> Any:
    """Return the underlying HuggingFace tokenizer from an mlx_lm TokenizerWrapper."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


def tokenize_rollouts(
    tokenizer: Any,
    rollout_texts: List[str],
    max_length: int,
    device: mx.Device = mx.cpu,
    padding_side: str = "right",
) -> Tuple[mx.array, mx.array]:
    if not rollout_texts:
        with mx.stream(mx.default_stream(device)):
            return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)

    hf_tok = _unwrap_tokenizer(tokenizer)
    orig_padding_side = getattr(hf_tok, "padding_side", "right")
    hf_tok.padding_side = padding_side

    batch = hf_tok(
        rollout_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=False,
        return_tensors=None,
    )

    hf_tok.padding_side = orig_padding_side

    with mx.stream(mx.default_stream(device)):
        return (
            mx.array(batch["input_ids"], dtype=mx.int32),
            mx.array(batch["attention_mask"], dtype=mx.int32),
        )


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

def parse_answer(text: str) -> Any:
    matches = re.findall(r"<answer>\s*([-+]?\d*\.?\d+)\s*<\/answer>", text, flags=re.IGNORECASE)
   
    # matches = re.findall(r"<summary>(.*?)</summary>", text, re.DOTALL)

    
    if matches:
        return float(matches[-1])

    final_answer_match = re.findall(
        r"final\s+answer[^\d\-\+]*([-+]?\d*\.?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return float(final_answer_match[-1])

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


# ---------------------------------------------------------------------------
# LoRA adapter application
# ---------------------------------------------------------------------------

def apply_lora_if_quantized(model: Any, config: Dict[str, Any]) -> bool:
    """Wrap linear layers with bfloat16 LoRA adapters.

    Applied automatically when the model contains uint32 (4-bit quantized) weights,
    or when `force_lora: true` is set in config (useful for bf16 models to reduce
    Freezes the base model and wraps transformer linear layers with trainable LoRA
    adapters. Returns True if LoRA was applied, False otherwise.
    """
    from mlx_lm.tuner.utils import linear_to_lora_layers

    flat = tree_flatten(model.parameters())
    dtypes: Dict[str, int] = {}
    for _, v in flat:
        key = str(v.dtype)
        dtypes[key] = dtypes.get(key, 0) + 1

    has_quantized = mx.uint32 in {v.dtype for _, v in flat}
    force_lora = bool(config.get("force_lora", False))

    sep = "=" * 60
    logger.info(sep)
    logger.info("[MODEL] Weight dtype distribution:")
    for dtype_name, count in sorted(dtypes.items()):
        logger.info("  %s: %d tensors", dtype_name, count)

    if not has_quantized and not force_lora:
        all_params = tree_flatten(model.parameters())
        n_total = sum(v.size for _, v in all_params)
        logger.info("[LORA] NOT applied — no uint32 weights and force_lora=false")
        logger.info("[LORA] All %d parameters are trainable (%.1f M)", n_total, n_total / 1e6)
        logger.info(sep)
        return False

    rank = int(config.get("lora_rank", 8))
    scale = float(config.get("lora_scale", 20.0))
    lora_cfg = {
        "rank": rank,
        "alpha": scale * rank,
        "dropout": float(config.get("lora_dropout", 0.0)),
        "scale": scale,
    }
    num_layers = int(config.get("lora_num_layers", -1))

    reason = "uint32 weights detected" if has_quantized else "force_lora=true"
    logger.info("[LORA] %s — applying LoRA adapters", reason)
    logger.info("[LORA] Config: rank=%d  scale=%.1f  dropout=%.1f  layers=%s",
                rank, scale, lora_cfg["dropout"], "all" if num_layers == -1 else num_layers)
    logger.info("[LORA] Freezing base model weights ...")

    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_cfg)

    trainable = tree_flatten(model.trainable_parameters())
    n_trainable = sum(v.size for _, v in trainable)
    all_params = tree_flatten(model.parameters())
    n_total = sum(v.size for _, v in all_params)

    logger.info("[LORA] ACTIVE — %d trainable params (%.1f M) out of %.1f M total (%.1f%% of model)",
                n_trainable, n_trainable / 1e6, n_total / 1e6,
                100.0 * n_trainable / n_total)
    logger.info("[LORA] Base backbone frozen; only lora_a / lora_b will update")
    logger.info(sep)
    return True
