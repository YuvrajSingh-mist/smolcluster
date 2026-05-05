"""Generic MLX training utilities for GRPO — gradient helpers, tokenisation, data batching."""

import logging
import os
import queue
import random
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
import re

import mlx.core as mx
import numpy as np
from mlx.nn.utils import checkpoint as mlx_grad_checkpoint
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load as mlx_load
from mlx_lm.utils import save_config as mlx_save_config

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set Python/NumPy/MLX RNG seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


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


def build_completion_mask(
    tokenizer: Any,
    rollout_questions: List[str],
    flat_mask: mx.array,
    num_prompts: int,
    num_rollouts: int,
) -> mx.array:
    """Return [T*C, D] mask that is 1 only for completion tokens."""
    hf_tok = _unwrap_tokenizer(tokenizer)
    prompt_lens_flat: List[int] = []
    for prompt_index in range(num_prompts):
        prompt_len = len(hf_tok.encode(rollout_questions[prompt_index * num_rollouts], add_special_tokens=False))
        prompt_lens_flat.extend([prompt_len] * num_rollouts)

    seq_len = flat_mask.shape[1]
    prompt_lens = mx.array(prompt_lens_flat, dtype=mx.int32)
    positions = mx.arange(seq_len, dtype=mx.int32)
    completion_mask = (positions[None, :] >= prompt_lens[:, None]).astype(mx.int32) * flat_mask
    mx.eval(completion_mask)
    return completion_mask


def filter_to_uniform_groups(
    per_prompt: List[Tuple[List[str], str]],
    num_rollouts: int,
    log: Optional[logging.Logger] = None,
) -> List[Tuple[List[str], str]]:
    """Keep only prompts that produced exactly num_rollouts completions."""
    active_logger = log or logger
    filtered: List[Tuple[List[str], str]] = []
    for index, (texts, answer) in enumerate(per_prompt):
        if len(texts) == 0:
            active_logger.warning("filter_to_uniform_groups: prompt %d produced 0 rollouts - dropping", index)
        elif len(texts) < num_rollouts:
            active_logger.warning(
                "filter_to_uniform_groups: prompt %d has %d/%d rollouts - dropping",
                index,
                len(texts),
                num_rollouts,
            )
        else:
            filtered.append((texts[:num_rollouts], answer))
    return filtered


def compute_advantages(rewards: mx.array, dtype: type = mx.float32) -> mx.array:
    """Normalize rewards into advantages within each prompt's rollout group."""
    group_mean = mx.mean(rewards, axis=1, keepdims=True)
    group_var = mx.mean((rewards - group_mean) ** 2, axis=1, keepdims=True)
    group_std = mx.sqrt(group_var)
    return (rewards - group_mean) / mx.maximum(group_std, mx.array(1e-6, dtype=dtype))


def compute_grpo_loss(
    curr_logprobs: mx.array,
    advantages: mx.array,
    config: Dict[str, Any],
    old_logprobs: Optional[mx.array] = None,
    ref_logprobs: Optional[mx.array] = None,
) -> mx.array:
    """GRPO macro-averaged loss."""
    if old_logprobs is not None:
        logprobs_ratio = mx.exp(curr_logprobs - old_logprobs)
        clipped_ratio = mx.clip(
            logprobs_ratio,
            1 - config["clip_ratio"],
            1 + config["clip_ratio"],
        )
        per_rollout = mx.minimum(logprobs_ratio * advantages, clipped_ratio * advantages)
    else:
        per_rollout = advantages * curr_logprobs

    if ref_logprobs is not None and config.get("use_kl", True):
        kl = mx.exp(ref_logprobs - curr_logprobs) - (ref_logprobs - curr_logprobs) - 1
        per_rollout = per_rollout - config["kl_beta"] * kl

    per_group = mx.mean(per_rollout, axis=1)
    return -mx.mean(per_group)


def compute_logprobs(
    model: Any,
    input_ids: mx.array,
    attention_mask: mx.array,
    dtype: type = mx.float32,
    use_checkpoint: bool = False,
    completion_mask: Optional[mx.array] = None,
) -> mx.array:
    """Compute per-sequence mean log-probs over completion tokens."""
    num_prompts, num_rollouts, seq_len = input_ids.shape
    flat_batch = num_prompts * num_rollouts
    flat_ids = input_ids.reshape(flat_batch, seq_len)
    flat_mask = attention_mask.reshape(flat_batch, seq_len)

    score_flat = completion_mask.reshape(flat_batch, seq_len) if completion_mask is not None else flat_mask
    shift_mask = score_flat[:, 1:]

    def _forward(ids: mx.array) -> mx.array:
        logits = model(ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        target_logit = mx.take_along_axis(shift_logits, shift_labels[..., None], axis=-1).squeeze(-1)
        log_z = mx.logsumexp(shift_logits, axis=-1)
        return target_logit - log_z

    if use_checkpoint:
        token_logprobs = mlx_grad_checkpoint(model, _forward)(flat_ids)
    else:
        token_logprobs = _forward(flat_ids)

    filtered_logprobs = mx.where(shift_mask > 0, token_logprobs, 0.0)
    counts = mx.maximum(mx.sum(shift_mask, axis=1), mx.array(1.0, dtype=dtype))
    flat_result = mx.sum(filtered_logprobs, axis=1) / counts
    return flat_result.reshape(num_prompts, num_rollouts)


def compute_ratio_stats(
    curr_logprobs: mx.array,
    ref_logprobs: mx.array,
    config: Dict[str, Any],
    dtype: type = mx.float32,
) -> Dict[str, float]:
    """Compute ratio/clip/KL diagnostics given already-computed logprob vectors."""
    ratio = mx.exp(curr_logprobs - ref_logprobs)
    lo = 1.0 - float(config["clip_ratio"])
    hi = 1.0 + float(config["clip_ratio"])
    clipped = mx.logical_or(ratio < lo, ratio > hi)
    clip_frac = float(mx.mean(clipped.astype(dtype)).item())
    kl = mx.exp(ref_logprobs - curr_logprobs) - (ref_logprobs - curr_logprobs) - 1
    mx.eval(ratio, kl)
    return {
        "ratio_mean": float(mx.mean(ratio).item()),
        "clip_frac": clip_frac,
        "kl_mean": float(mx.mean(kl).item()),
    }


class RolloutPrefetcher:
    """Fetch the next step's rollouts in a background thread while compute runs."""

    def __init__(
        self,
        fetch_fn: Callable[[List[str], List[str], Optional[int]], List[Tuple[List[str], str]]],
    ) -> None:
        self._fetch_fn = fetch_fn
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None

    def submit(self, prompts: List[str], answers: List[str], step: Optional[int] = None) -> None:
        import time as _time
        _start = _time.monotonic()

        def _run() -> None:
            result = self._fetch_fn(prompts, answers, step)
            elapsed = _time.monotonic() - _start
            self._queue.put((result, elapsed))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get(self) -> Tuple[List[Tuple[List[str], str]], float]:
        """Return ``(rollouts, rollout_time_s)`` — blocks until the background fetch finishes."""
        return self._queue.get()

    def flush(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=120)
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass


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
    """Parse numeric answer from <answer>...</answer> tags.
    Handles comma/space-separated thousands (e.g., '25,000' or '1 234').
    """
    # Extract content within <answer> tags
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return float("nan")
    
    answer_text = match.group(1).strip()
    
    # Normalize: remove commas and spaces (for thousands separators)
    # But validate we're not removing actual content
    normalized = answer_text.replace(",", "").replace(" ", "")
    
    # Reject if contains alphabetic characters (likely spurious text)
    if re.search(r"[a-zA-Z]", normalized):
        return float("nan")
    
    # Try to parse as float
    try:
        return float(normalized)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    dtype: type,
    config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Tuple[Any, Optional[Any], Any, Dict[str, Any]]:
    """Load the policy model (and optionally the reference model) from HF.

    Args:
        dtype:        MLX dtype for the model (e.g. mx.bfloat16).
        config:       GRPO training config dict.
        model_config: Model config dict (contains ``dp.hf_model_name``).

    Returns:
        (model, ref_model, tokenizer, model_cfg) — ref_model is None when
        use_kl=false; model_cfg is the raw HF config dict (includes quantization
        metadata if the base model is quantized).
    """
    model_name = model_config["dp"]["hf_model_name"]
    tokenizer_config = {
        "trust_remote_code": True if config.get("trust_remote_code", False) else None
    }

    device = get_mlx_device(config)
    device_stream = mx.default_stream(device)
    logger.info("Loading MLX model: %s (device=%s)", model_name, config.get("device", "cpu"))

    model, tokenizer, model_cfg = mlx_load(model_name, tokenizer_config=tokenizer_config, return_config=True)
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

    return model, ref_model, tokenizer, model_cfg


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
