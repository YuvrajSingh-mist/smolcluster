"""Generic MLX training utilities for GRPO — gradient helpers, tokenisation, data batching."""

import json
import logging
import os
import queue
import random
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.nn.utils import checkpoint as mlx_grad_checkpoint
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load as mlx_load

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

def get_dtype_from_config(config: dict[str, Any]) -> type:
    """Map config dtype string ('float32', 'bfloat16') to an MLX dtype."""
    dtype_str = str(config.get("dtype", "float32")).lower()
    if dtype_str == "bfloat16":
        return mx.bfloat16
    if dtype_str == "float32":
        return mx.float32
    logger.warning("Unknown dtype '%s', defaulting to float32", dtype_str)
    return mx.float32


def get_mlx_device(config: dict[str, Any]) -> mx.Device:
    """Return the MLX Device for config['device'] ('cpu', 'gpu', or 'metal')."""
    device_str = str(config.get("device", "cpu")).lower()
    return mx.gpu if device_str in ("gpu", "metal") else mx.cpu


# ---------------------------------------------------------------------------
# Data batching
# ---------------------------------------------------------------------------

def iterate_batches(
    examples: Sequence[tuple[str, str]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[list[str], list[str]]]:
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
    rollout_texts: list[str],
    max_length: int,
    device: mx.Device = mx.cpu,
    padding_side: str = "right",
) -> tuple[mx.array, mx.array]:
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
    rollout_questions: list[str],
    flat_mask: mx.array,
    num_prompts: int,
    num_rollouts: int,
) -> mx.array:
    """Return [T*C, D] mask that is 1 only for completion tokens."""
    hf_tok = _unwrap_tokenizer(tokenizer)
    prompt_lens_flat: list[int] = []
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
    per_prompt: list[tuple[list[str], str]],
    num_rollouts: int,
    log: logging.Logger | None = None,
) -> list[tuple[list[str], str]]:
    """Keep only prompts that produced exactly num_rollouts completions."""
    active_logger = log or logger
    filtered: list[tuple[list[str], str]] = []
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
    config: dict[str, Any],
    old_logprobs: mx.array | None = None,
    ref_logprobs: mx.array | None = None,
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
    completion_mask: mx.array | None = None,
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
    config: dict[str, Any],
    dtype: type = mx.float32,
) -> dict[str, float]:
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
        fetch_fn: Callable[[list[str], list[str], int | None], list[tuple[list[str], str]]],
    ) -> None:
        self._fetch_fn = fetch_fn
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    def submit(self, prompts: list[str], answers: list[str], step: int | None = None) -> None:
        import time as _time
        _start = _time.monotonic()

        def _run() -> None:
            result = self._fetch_fn(prompts, answers, step)
            elapsed = _time.monotonic() - _start
            self._queue.put((result, elapsed))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get(self) -> tuple[list[tuple[list[str], str]], float]:
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
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[Any, Any | None, Any, dict[str, Any]]:
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

    ref_model: Any | None = None
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

def apply_lora_if_quantized(model: Any, config: dict[str, Any]) -> bool:
    """Wrap linear layers with bfloat16 LoRA adapters.

    Applied automatically when the model contains uint32 (4-bit quantized) weights,
    or when `force_lora: true` is set in config (useful for bf16 models to reduce
    Freezes the base model and wraps transformer linear layers with trainable LoRA
    adapters. Returns True if LoRA was applied, False otherwise.
    """
    from mlx_lm.tuner.utils import linear_to_lora_layers

    flat = tree_flatten(model.parameters())
    dtypes: dict[str, int] = {}
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


# ---------------------------------------------------------------------------
# Dashboard metrics helpers (shared by all GRPO training loops)
# ---------------------------------------------------------------------------

DASHBOARD_METRICS_FILE = Path("/tmp/smolcluster_metrics.json")
DASHBOARD_GRAD_PING = Path("/tmp/smolcluster_grad_ping")
DASHBOARD_GRAD_INTERVAL = Path("/tmp/smolcluster_grad_interval_ms")


def get_optimizer_lr(optimizer: Any, config: dict[str, Any]) -> float:
    """Return the current learning rate for different MLX optimizer wrappers."""
    lr_obj = getattr(optimizer, "learning_rate", None)
    if lr_obj is not None:
        try:
            return float(lr_obj.item()) if hasattr(lr_obj, "item") else float(lr_obj)
        except Exception:
            logger.info("Could not parse learning_rate from optimizer, falling back to config value")
    if hasattr(optimizer, "_lr"):
        try:
            return float(optimizer._lr)
        except Exception:
            logger.info("Could not parse _lr from optimizer, falling back to config value")
    return float(config.get("learning_rate", 0.0))


def publish_dashboard_metrics(
    metrics: dict,
    *,
    global_step: int,
    total_steps: int,
    grad_norm: float,
    lr: float,
    skipped_update: bool,
    last_ts: float | None,
    grove: Any = None,
    net_stats: dict[str, float] | None = None,
) -> float | None:
    """Write step metrics to the dashboard file and return the updated timestamp.

    Args:
        last_ts: The timestamp recorded at the previous gradient step (or None).

    Returns:
        The timestamp of this step (or the unchanged *last_ts* if the update was
        skipped), to be stored by the caller and passed back on the next call.
    """
    try:
        now = time.time()
        safe_grad_norm: Any = grad_norm
        if safe_grad_norm != safe_grad_norm or safe_grad_norm in (float("inf"), float("-inf")):
            safe_grad_norm = "NaN"
        step_time_s = None if last_ts is None else max(0.0, now - last_ts)
        est_throughput = None
        token_count = float(metrics.get("generation_token_len_mean", 0) or 0) * float(
            metrics.get("num_rollouts", 0) or 0
        )
        if step_time_s and step_time_s > 0.0 and token_count > 0.0:
            est_throughput = round(token_count / step_time_s, 1)
        rollout_time_s = float(metrics.get("rollout_time_s") or 0)
        _tbase = rollout_time_s if rollout_time_s > 0.0 else step_time_s
        prompt_count = float(metrics.get("prompt_token_len_mean", 0) or 0) * float(
            metrics.get("num_rollouts", 0) or 0
        )
        tok_sec_in = round(prompt_count / _tbase, 1) if (_tbase and prompt_count > 0.0) else est_throughput
        tok_sec_out = round(token_count / _tbase, 1) if (_tbase and token_count > 0.0) else est_throughput
        eta_remaining = None
        if step_time_s and step_time_s > 0.0 and total_steps and total_steps > 0:
            steps_left = max(0, int(total_steps) - int(global_step))
            eta_seconds = int(max(0.0, steps_left * step_time_s))
            h = eta_seconds // 3600
            m = (eta_seconds % 3600) // 60
            s = eta_seconds % 60
            eta_remaining = f"{h:02d}:{m:02d}:{s:02d}"
        _ns = net_stats or {}
        payload = {
            "step": global_step,
            "total_steps": total_steps,
            "loss": float(metrics.get("loss", 0.0)),
            "throughput": est_throughput,
            "tok_sec_in": tok_sec_in,
            "tok_sec_out": tok_sec_out,
            "grad_norm": safe_grad_norm,
            "lr": lr,
            "eta_remaining": eta_remaining,
            "algorithm": "grpo",
            "running": True,
            "reward": float(metrics.get("reward", 0.0)),
            "active_mem_mb": round(mx.metal.get_active_memory() / 1e6, 1) if mx.metal.is_available() else 0.0,
            "peak_mem_mb": round(mx.metal.get_peak_memory() / 1e6, 1) if mx.metal.is_available() else 0.0,
            **{k: v for k, v in _ns.items() if isinstance(v, (int, float))},
        }
        DASHBOARD_METRICS_FILE.write_text(json.dumps(payload))
        print(f"[SMOL_METRICS] {json.dumps(payload)}", flush=True)
        if not skipped_update:
            DASHBOARD_GRAD_PING.touch()
            if step_time_s and step_time_s > 0.0:
                DASHBOARD_GRAD_INTERVAL.write_text(str(round(step_time_s * 1000, 1)))
            last_ts = now
        if grove is not None and not skipped_update:
            _gn = grad_norm if isinstance(grad_norm, float) and grad_norm == grad_norm else None
            grove.report(
                float(metrics.get("loss", 0.0)),
                step=global_step,
                grad_norm=_gn,
                tok_per_sec=est_throughput,
            )
            grove.status("training")
    except Exception:
        logger.info("Exception occurred while reporting metrics to grove")
    return last_ts
