

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
import yaml
from tqdm.auto import tqdm

from smolcluster.applications.reasoning.grpo.data.summarization import build_train_val_examples
from smolcluster.applications.reasoning.grpo.rewards import (
    calculate_answer_reward,
    calculate_summary_quality,
    calculate_length_reward,
)
from smolcluster.applications.reasoning.grpo.utils.rollouts import build_batched_rollout_texts, build_rollouts_per_prompt
from smolcluster.applications.reasoning.grpo.utils import (
    _add_grads,
    _log_mem,
    _scale_grads,
    build_completion_mask,
    compute_advantages,
    compute_grpo_loss,
    compute_logprobs,
    compute_ratio_stats,
    filter_to_uniform_groups,
    get_dtype_from_config,
    get_mlx_device,
    iterate_batches,
    load_model,
    apply_lora_if_quantized,
    tokenize_rollouts,
    parse_answer,
    GradScaler,
    MasterWeightAdamW,
    RolloutPrefetcher,
    set_global_seed,
)
from smolcluster.applications.reasoning.grpo.utils.worker_sync import (
    save_policy_weights,
    sync_and_reload_workers,
)
from smolcluster.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


_module_dir = Path(__file__).parent
_smolcluster_root = _module_dir.parents[2]
grpo_config_path = _smolcluster_root / "configs" / "inference" / "reasoning" / "grpo" / "config.yaml"
model_config_path = _smolcluster_root / "configs" / "inference" / "model_config_inference.yaml"

with open(grpo_config_path) as f:
    grpo_config = yaml.safe_load(f)

with open(model_config_path) as f:
    model_config = yaml.safe_load(f)

_debug_dir = _module_dir.parents[4] / ".grpo_debug"
_debug_dir.mkdir(parents=True, exist_ok=True)
_answers_log_path = _debug_dir / "rollout_answers.jsonl"
_answers_log_lock = threading.Lock()
_DASHBOARD_METRICS_FILE = Path("/tmp/smolcluster_metrics.json")
_DASHBOARD_GRAD_PING = Path("/tmp/smolcluster_grad_ping")
_DASHBOARD_GRAD_INTERVAL = Path("/tmp/smolcluster_grad_interval_ms")
_LAST_DASHBOARD_GRAD_TS: Optional[float] = None


MAX_LENGTH_OF_SUMMARIZATION = 64

def _append_answers_log(record: dict) -> None:
    with _answers_log_lock:
        with _answers_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")


def _publish_dashboard_metrics(metrics: dict, *, global_step: int, total_steps: int,
                               grad_norm: float, lr: float, skipped_update: bool) -> None:
    global _LAST_DASHBOARD_GRAD_TS
    try:
        now = time.time()
        safe_grad_norm = grad_norm
        if safe_grad_norm != safe_grad_norm or safe_grad_norm in (float("inf"), float("-inf")):
            safe_grad_norm = "NaN"
        step_time_s = None if _LAST_DASHBOARD_GRAD_TS is None else max(0.0, now - _LAST_DASHBOARD_GRAD_TS)
        est_throughput = None
        token_count = float(metrics.get("generation_token_len_mean", 0) or 0) * float(metrics.get("num_rollouts", 0) or 0)
        if step_time_s and step_time_s > 0.0 and token_count > 0.0:
            est_throughput = round(token_count / step_time_s, 1)
        # Use the measured vLLM generation time for accurate prefill / decode tok/sec.
        # Prefill (input) = prompt tokens processed per second by vLLM.
        # Decode (output) = new tokens generated per second by vLLM.
        # Fall back to step_time_s if rollout_time_s was not captured (prefetch path).
        rollout_time_s = float(metrics.get("rollout_time_s") or 0)
        _tbase = rollout_time_s if rollout_time_s > 0.0 else step_time_s
        prompt_count = float(metrics.get("prompt_token_len_mean", 0) or 0) * float(metrics.get("num_rollouts", 0) or 0)
        tok_sec_in  = round(prompt_count / _tbase, 1) if (_tbase and prompt_count > 0.0) else est_throughput
        tok_sec_out = round(token_count   / _tbase, 1) if (_tbase and token_count  > 0.0) else est_throughput
        eta_remaining = None
        if step_time_s and step_time_s > 0.0 and total_steps and total_steps > 0:
            steps_left = max(0, int(total_steps) - int(global_step))
            eta_seconds = int(max(0.0, steps_left * step_time_s))
            h = eta_seconds // 3600
            m = (eta_seconds % 3600) // 60
            s = eta_seconds % 60
            eta_remaining = f"{h:02d}:{m:02d}:{s:02d}"

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
        }
        _DASHBOARD_METRICS_FILE.write_text(json.dumps(payload))
        print(f"[SMOL_METRICS] {json.dumps(payload)}", flush=True)
        if not skipped_update:
            _DASHBOARD_GRAD_PING.touch()
            if step_time_s and step_time_s > 0.0:
                _DASHBOARD_GRAD_INTERVAL.write_text(str(round(step_time_s * 1000, 1)))
            _LAST_DASHBOARD_GRAD_TS = now
    except Exception:
        pass


def _optimizer_lr(optimizer: Any, config: Dict[str, Any]) -> float:
    """Return current LR for different optimizer wrappers without assuming attributes."""
    lr_obj = getattr(optimizer, "learning_rate", None)
    if lr_obj is not None:
        try:
            return float(lr_obj.item()) if hasattr(lr_obj, "item") else float(lr_obj)
        except Exception:
            pass
    if hasattr(optimizer, "_lr"):
        try:
            return float(getattr(optimizer, "_lr"))
        except Exception:
            pass
    return float(config.get("learning_rate", 0.0))


def _compute_single_reward(
    args: Tuple[int, str, str, str, Any, bool, bool, bool, bool],
) -> Tuple[int, float, dict]:
    idx, question, generated_text, true_answer, tokenizer, use_rouge, use_meteor, use_bleu, use_length_penalty = args
    quality_scores = calculate_summary_quality(
        generated_text, true_answer,
        use_rouge=use_rouge, use_meteor=use_meteor, use_bleu=use_bleu,
    )
    length_penalty = calculate_length_reward(generated_text, MAX_LENGTH_OF_SUMMARIZATION, tokenizer=tokenizer) if use_length_penalty else 0.0
    total_reward = float(sum(quality_scores.values()) + length_penalty)
    log_record = {
        "rollout_idx":    idx,
        "question":       question,
        **{f"quality_{k}": v for k, v in quality_scores.items()},
        "length_penalty": float(length_penalty),
        "total_reward":   total_reward,
        "generated_text": generated_text,
        "true_answer":    true_answer,
    }
    return idx, total_reward, log_record


def compute_rewards(
    rollout_texts: List[str],
    rollout_targets: List[str],
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    max_workers: Optional[int] = None,
    step: Optional[int] = None,
    rollout_questions: Optional[List[str]] = None,
    tokenizer: Optional[Any] = None,
    use_rouge: bool = False,
    use_meteor: bool = False,
    use_bleu: bool = False,
    use_length_penalty: bool = True,
) -> Tuple[mx.array, Dict[str, List[float]]]:
    """Returns (reward_tensor [T*C], components) where components has per-rollout
    lists for each enabled quality metric plus length_penalty and total_reward."""
    questions = rollout_questions if rollout_questions is not None else [""] * len(rollout_texts)
    indexed_args = [
        (i, q, text, target, tokenizer, use_rouge, use_meteor, use_bleu, use_length_penalty)
        for i, (q, text, target) in enumerate(zip(questions, rollout_texts, rollout_targets))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_single_reward, indexed_args))

    reward_values: List[float] = []
    log_records: List[dict] = []
    for _idx, total_reward, log_record in results:
        reward_values.append(total_reward)
        log_records.append(log_record)

    _append_answers_log({"step": step, "rollouts": log_records})

    # Union of quality keys across all records (safe regardless of which metrics are on)
    quality_keys = sorted({k for r in log_records for k in r if k.startswith("quality_")})
    components: Dict[str, List[float]] = {
        **{k: [r[k] for r in log_records] for k in quality_keys},
        "length_penalty": [r["length_penalty"] for r in log_records],
        "total_reward":   [r["total_reward"]   for r in log_records],
    }

    with mx.stream(mx.default_stream(device)):
        return mx.array(reward_values, dtype=dtype), components


def evaluate_batch(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    prompts: List[str],
    true_answers: List[str],
    ref_model: Optional[Any] = None,
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    step: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    per_prompt_raw = build_rollouts_per_prompt(prompts, true_answers, config, step=step)
    effective_rollouts = len(per_prompt_raw[0][0]) if per_prompt_raw and per_prompt_raw[0][0] else int(config["num_rollouts"])
    per_prompt = filter_to_uniform_groups(per_prompt_raw, effective_rollouts)
    if not per_prompt:
        return None

    T = len(per_prompt)
    C = effective_rollouts
    rollout_texts: List[str] = []
    rollout_targets: List[str] = []
    rollout_questions: List[str] = []
    for (texts, ans), prompt in zip(per_prompt, prompts[:T]):
        rollout_texts.extend(texts)
        rollout_targets.extend([ans] * C)
        rollout_questions.extend([prompt] * C)

    full_texts = [q + t for q, t in zip(rollout_questions, rollout_texts)]
    flat_ids, flat_mask = tokenize_rollouts(tokenizer, full_texts, config["max_input_tokens"], device=device)
    D = flat_ids.shape[1]
    input_ids      = flat_ids.reshape(T, C, D)   # [T, C, D]
    attention_mask = flat_mask.reshape(T, C, D)
    completion_mask = build_completion_mask(tokenizer, rollout_questions, flat_mask, T, C).reshape(T, C, D)

    model.eval()
    old_logprobs = compute_logprobs(model, input_ids, attention_mask, dtype=dtype, completion_mask=completion_mask)  # [T, C]
    mx.eval(old_logprobs)
    model.train()

    curr_logprobs = compute_logprobs(model, input_ids, attention_mask, dtype=dtype, completion_mask=completion_mask)  # [T, C]

    _qm = config.get("quality_metrics", {})
    rewards_flat, reward_components = compute_rewards(
        rollout_texts,
        rollout_targets,
        dtype=dtype,
        device=device,
        max_workers=config.get("reward_workers"),
        step=step,
        rollout_questions=rollout_questions,
        tokenizer=tokenizer,
        use_rouge=bool(_qm.get("rouge", False)),
        use_meteor=bool(_qm.get("meteor", False)),
        use_bleu=bool(_qm.get("bleu", False)),
        use_length_penalty=bool(_qm.get("length_penalty", True)),
    )
    rewards = rewards_flat.reshape(T, C)          # [T, C]
    advantages = compute_advantages(rewards, dtype=dtype)  # [T, C]

    ref_logprobs: Optional[mx.array] = None
    if ref_model is not None and config.get("use_kl", True):
        ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask, dtype=dtype, completion_mask=completion_mask)  # [T, C]

    loss = compute_grpo_loss(
        curr_logprobs, advantages, config,
        old_logprobs=old_logprobs, ref_logprobs=ref_logprobs,
    )
    mx.eval(loss, rewards)

    return {
        "loss": float(loss.item()),
        "reward": float(mx.mean(rewards).item()),
        "num_rollouts": float(T * C),
    }


def train_step(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    prompts: List[str],
    true_answers: List[str],
    epoch_idx: int,
    step_in_epoch: int,
    total_steps_in_epoch: int,
    ref_model: Optional[Any] = None,
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    device_stream: Optional[mx.Stream] = None,
    scaler: Optional[GradScaler] = None,
    prefetched_rollouts: Optional[Tuple[List[str], List[str], List[str]]] = None,
    rollout_group_sizes: Optional[List[int]] = None,
    
    step: Optional[int] = None,
) -> Optional[Tuple[Dict[str, float], Any]]:
    step_tag = f"[train_step epoch={epoch_idx + 1} step={step_in_epoch}/{total_steps_in_epoch}]"

    num_rollouts = int(config["num_rollouts"])

    if prefetched_rollouts is not None:
        # Reconstruct per_prompt from flat lists + group_sizes so we can apply
        # uniform-group filtering consistently with the non-prefetch path.
        flat_texts, flat_targets, flat_questions = prefetched_rollouts
        per_prompt_raw: List[Tuple[List[str], str]] = []
        idx = 0
        for size in (rollout_group_sizes or []):
            per_prompt_raw.append((flat_texts[idx : idx + size], flat_targets[idx] if idx < len(flat_targets) else ""))
            idx += size
        logger.info("\n%s Using prefetched rollouts (%d)", step_tag, len(flat_texts))
        rollout_time_s: Optional[float] = None  # prefetched while previous step was running
    else:
        logger.info("\n%s Generating rollouts for %d prompt(s) ...", step_tag, len(prompts))
        _t0 = time.time()
        per_prompt_raw = build_rollouts_per_prompt(prompts, true_answers, config, step=step)
        rollout_time_s = max(0.001, time.time() - _t0)
        logger.info("%s [TIMING] rollout_gen: %.1fs", step_tag, rollout_time_s)

    # Infer effective C from actual data: num_rollouts_per_worker × num_workers.
    # config["num_rollouts"] is per-worker; each prompt receives that many from EACH worker.
    effective_rollouts = len(per_prompt_raw[0][0]) if per_prompt_raw and per_prompt_raw[0][0] else num_rollouts

    # Enforce uniform C: drop prompts that didn't get exactly effective_rollouts completions.
    per_prompt = filter_to_uniform_groups(per_prompt_raw, effective_rollouts)
    if not per_prompt:
        logger.warning("%s No prompts with full rollouts after filtering, skipping step", step_tag)
        return None

    T = len(per_prompt)
    C = effective_rollouts
    rollout_texts: List[str] = []
    rollout_targets: List[str] = []
    rollout_questions: List[str] = []
    for (texts, ans), prompt in zip(per_prompt, prompts[:T]):
        rollout_texts.extend(texts)
        rollout_targets.extend([ans] * C)
        rollout_questions.extend([prompt] * C)
    logger.info("%s Got %d prompt(s) × %d rollouts = %d total", step_tag, T, C, T * C)

    # Tokenize prompt+completion together so the model sees full context.
    # completion_mask zeros out prompt positions so log-prob averaging is over
    # generated tokens only, giving p(completion | prompt).
    full_texts = [q + t for q, t in zip(rollout_questions, rollout_texts)]
    flat_ids, flat_mask = tokenize_rollouts(tokenizer, full_texts, config["max_input_tokens"], device=device)
    D = flat_ids.shape[1]
    input_ids      = flat_ids.reshape(T, C, D)    # [T, C, D]
    attention_mask = flat_mask.reshape(T, C, D)   # [T, C, D]
    completion_mask = build_completion_mask(tokenizer, rollout_questions, flat_mask, T, C).reshape(T, C, D)
    logger.info("%s input_ids shape: %s  (prompt+completion, D=%d)", step_tag, list(input_ids.shape), D)
    _log_mem("train_step: after tokenize_rollouts")

    logger.info("%s Computing rewards ...", step_tag)
    _qm = config.get("quality_metrics", {})
    rewards_flat, reward_components = compute_rewards(
        rollout_texts,
        rollout_targets,
        dtype=dtype,
        device=device,
        max_workers=config.get("reward_workers"),
        step=step,
        rollout_questions=rollout_questions,
        tokenizer=tokenizer,
        use_rouge=bool(_qm.get("rouge", False)),
        use_meteor=bool(_qm.get("meteor", False)),
        use_bleu=bool(_qm.get("bleu", False)),
        use_length_penalty=bool(_qm.get("length_penalty", True)),
    )
    rewards = rewards_flat.reshape(T, C)          # [T, C]
    logger.info(
        "%s Rewards - mean: %.4f  min: %.4f  max: %.4f",
        step_tag,
        float(mx.mean(rewards).item()),
        float(mx.min(rewards).item()),
        float(mx.max(rewards).item()),
    )
    logger.info("%s Computing advantages ...", step_tag)
    advantages = compute_advantages(rewards, dtype=dtype)         # [T, C]
    rewards_std_value = float(mx.std(rewards).item())
    if rewards_std_value < 1e-8:
        logger.warning(
            "%s Reward std is ~0 (%.3e). Advantages collapse and grad_norm may be 0.",
            step_tag,
            rewards_std_value,
        )

    # old_logprobs: snapshot of the policy that generated the rollouts.
    # Computed from current model weights (same weights vLLM was synced to)
    # BEFORE any gradient update this step. model.eval() disables dropout,
    # matching vLLM inference behaviour. No extra model copy needed.
    logger.info("%s Computing old policy log-probs ...", step_tag)
    model.eval()
    old_logprobs = compute_logprobs(model, input_ids, attention_mask, dtype=dtype, completion_mask=completion_mask)  # [T, C]
    mx.eval(old_logprobs)
    model.train()

    # ref_logprobs: frozen initial model, used only for KL penalty (use_kl=True).
    ref_logprobs: Optional[mx.array] = None
    if ref_model is not None and config.get("use_kl", True):
        logger.info("%s Computing reference model log-probs ...", step_tag)
        ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask, dtype=dtype, completion_mask=completion_mask)  # [T, C]
        mx.eval(ref_logprobs)

    use_ckpt = config.get("grad_checkpoint", False)
    # Chunk over T (prompts) and C (rollouts) so each backward sees only a small batch.
    chunk_T = int(config.get("grad_chunk_size", T))
    rollout_chunk = int(config.get("rollout_grad_chunk", C))

    mx.reset_peak_memory()
    _log_mem("train_step: before grad computation (peak reset)")
    logger.info(
        "%s Computing GRPO loss and gradients (T=%d C=%d chunk_T=%d rollout_chunk=%d) ...",
        step_tag, T, C, chunk_T, rollout_chunk,
    )

    accum_grads = None
    total_loss = 0.0
    _t_grad = time.time()

    # Store curr_logprobs for ratio diagnostics (avoid recomputing after grad loop)
    curr_lps: List[mx.array] = []

    for t_start in range(0, T, chunk_T):
        t_end = min(t_start + chunk_T, T)

        for rc_start in range(0, C, rollout_chunk):
            rc_end = min(rc_start + rollout_chunk, C)
            # Scale: each chunk contributes (chunk_T × chunk_C) / (T × C) of the total loss.
            # compute_grpo_loss computes mean over its T and C dims, so multiplying by
            # chunk_scale converts that mean into the correct sum contribution: sum / (T*C).
            chunk_scale = (t_end - t_start) * (rc_end - rc_start) / (T * C)

            c_ids  = input_ids[t_start:t_end, rc_start:rc_end]       # [chunk_T, chunk_C, D]
            c_mask = attention_mask[t_start:t_end, rc_start:rc_end]
            c_comp = completion_mask[t_start:t_end, rc_start:rc_end]
            c_adv  = advantages[t_start:t_end, rc_start:rc_end]       # [chunk_T, chunk_C]
            c_old  = old_logprobs[t_start:t_end, rc_start:rc_end]     # [chunk_T, chunk_C]
            c_ref  = ref_logprobs[t_start:t_end, rc_start:rc_end] if ref_logprobs is not None else None

            # Capture loop variables by value via default args to avoid closure bugs
            def chunk_loss_fn(m, _ids=c_ids, _mask=c_mask, _comp=c_comp,
                              _adv=c_adv, _old=c_old, _ref=c_ref) -> mx.array:
                lp = compute_logprobs(m, _ids, _mask, dtype=dtype, use_checkpoint=use_ckpt, completion_mask=_comp)  # [chunk_T, chunk_C]
                curr_lps.append(lp)
                chunk_loss = compute_grpo_loss(
                    lp, _adv, config, old_logprobs=_old, ref_logprobs=_ref
                ) * chunk_scale
                if scaler is not None:
                    return scaler.scale_loss(chunk_loss)
                return chunk_loss

            chunk_loss, chunk_grads = nn.value_and_grad(model, chunk_loss_fn)(model)
            with mx.stream(device_stream):
                mx.eval(chunk_loss, chunk_grads)
            _log_mem(f"train_step: chunk t[{t_start}:{t_end}] rc[{rc_start}:{rc_end}] after eval")

            chunk_loss_value = float(chunk_loss.item())
            if scaler is not None and scaler.enabled:
                chunk_loss_value /= scaler.get_scale()
            total_loss += chunk_loss_value
            if accum_grads is None:
                accum_grads = chunk_grads
            else:
                accum_grads = _add_grads(accum_grads, chunk_grads)
                with mx.stream(device_stream):
                    mx.eval(accum_grads)

    logger.info("%s Loss: %.6f  [TIMING] grad_loop: %.1fs", step_tag, total_loss, time.time() - _t_grad)

    # ---- extra diagnostics (no grad needed) --------------------------------
    # completion_mask is [T, C, D]; sum over token dim → per-rollout completion lengths [T, C]
    lengths = mx.sum(completion_mask, axis=-1).astype(dtype)
    lp = mx.array(reward_components["length_penalty"]) if reward_components else None
    metrics_extra: Dict[str, float] = {
        "reward_std":       float(mx.std(rewards).item()),
        "reward_min":       float(mx.min(rewards).item()),
        "reward_max":       float(mx.max(rewards).item()),
        "advantage_mean":   float(mx.mean(advantages).item()),
        "advantage_std":    float(mx.std(advantages).item()),
        "generation_token_len_mean": float(mx.mean(lengths).item()),
        "generation_token_len_min":  float(mx.min(lengths).item()),
        "generation_token_len_max":  float(mx.max(lengths).item()),
        "prompt_token_len_mean":     max(0.0, D - float(mx.mean(lengths).item())),
        "rollout_time_s":            rollout_time_s if rollout_time_s is not None else 0.0,
        "length_penalty_mean": float(mx.mean(lp).item()) if lp is not None else 0,
        "length_penalty_min":  float(mx.min(lp).item())  if lp is not None else 0,
        "length_penalty_max":  float(mx.max(lp).item())  if lp is not None else 0,
    }
    # Per-metric quality stats (rouge_l, meteor, bleu — whatever was enabled)
    if reward_components:
        for key in [k for k in reward_components if k.startswith("quality_")]:
            arr = mx.array(reward_components[key])
            metrics_extra[f"{key}_mean"] = float(mx.mean(arr).item())
            metrics_extra[f"{key}_std"]  = float(mx.std(arr).item())
            metrics_extra[f"{key}_min"]  = float(mx.min(arr).item())
            metrics_extra[f"{key}_max"]  = float(mx.max(arr).item())
    # Ratio stats: curr vs old policy (always available now)
    curr_lps_flat = mx.concatenate([lp.reshape(-1) for lp in curr_lps])  # [T*C]
    old_flat = old_logprobs.reshape(-1)                             # [T*C]
    metrics_extra.update(compute_ratio_stats(curr_lps_flat, old_flat, config, dtype))
    # -------------------------------------------------------------------------

    return {
        "loss": total_loss,
        "reward": float(mx.mean(rewards).item()),
        "num_rollouts": float(len(rollout_texts)),
        **metrics_extra,
        "_reward_components": reward_components,
    }, accum_grads


def validate(
    model: Any,
    config: Dict[str, Any],
    val_examples: Sequence[Tuple[str, str]],
    tokenizer: Any,
    global_step: int,
    ref_model: Optional[Any] = None,
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
) -> Dict[str, float]:
    model.eval()

    loss_values: List[float] = []
    reward_values: List[float] = []
    rollout_counts: List[float] = []

    logger.info("Running evaluation at step %s", global_step)
    for prompts, true_answers in iterate_batches(
        val_examples,
        batch_size=config["batch_size"],
        shuffle=False,
        seed=global_step,
    ):
        metrics = evaluate_batch(
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompts=prompts,
            true_answers=true_answers,
            ref_model=ref_model,
            dtype=dtype,
            device=device,
            step=global_step,
        )
        if metrics is None:
            logger.warning("No metrics were returned. Skipping logging of metrics for this batch.")
            continue
        loss_values.append(metrics["loss"])
        reward_values.append(metrics["reward"])
        rollout_counts.append(metrics["num_rollouts"])

    model.train()

    if not loss_values:
        return {
            "val/loss": 0.0,
            "val/reward": 0.0,
            "val/num_rollouts": 0.0,
        }

    return {
        "val/loss": sum(loss_values) / len(loss_values),
        "val/reward": sum(reward_values) / len(reward_values),
        "val/num_rollouts": sum(rollout_counts) / len(rollout_counts),
    }


def train(
    model: Any,
    config: Dict[str, Any],
    train_examples: Sequence[Tuple[str, str]],
    val_examples: Sequence[Tuple[str, str]],
    optimizer: Any,
    tokenizer: Any,
    ref_model: Optional[Any] = None,
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    scaler: Optional[GradScaler] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    model.train()
  
    global_step = 0
    total_epochs = config["num_epochs"]
    val_steps = config["val_steps"]
    base_seed = int(config.get("seed", 42))
    save_every_steps = int(config.get("weight_sync", {}).get("save_every_steps", 0))
    sync_steps = int(config.get("weight_sync", {}).get("sync_steps", 0))
    checkpoint_dir = str(config.get("weight_sync", {}).get("checkpoint_dir", "checkpoints/grpo"))

    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    use_prefetch = bool(config.get("prefetch_rollouts", False))
    logger.info(
        "Starting MLX GRPO training for %s epochs (grad_accum_steps=%d, prefetch=%s)",
        total_epochs, grad_accum_steps, use_prefetch,
    )
    rollout_step = 0
    train_start_time = time.time()

    for epoch in range(total_epochs):
        epoch_batches = list(
            iterate_batches(
                train_examples,
                batch_size=config["batch_size"],
                shuffle=True,
                seed=base_seed + epoch,
            )
        )
        epoch_bar = tqdm(
            epoch_batches,
            desc=f"epoch {epoch + 1}/{total_epochs}",
            leave=True,
        )

        total_steps_in_epoch = len(epoch_batches)

        prefetcher = (
            RolloutPrefetcher(
                lambda batch_prompts, batch_answers, step: build_rollouts_per_prompt(
                    batch_prompts,
                    batch_answers,
                    config,
                    step=step,
                )
            )
            if use_prefetch
            else None
        )
        if prefetcher and epoch_batches:
            prefetcher.submit(*epoch_batches[0], step=rollout_step + 1)
            logger.info("Prefetcher armed for epoch %d step 0.", epoch + 1)

        for step_in_epoch, (prompts, true_answers) in enumerate(epoch_bar, start=0):
            current_rollout_step = rollout_step + 1
            rollout_step = current_rollout_step
            # Collect prefetched rollouts (blocks only on cold-start or very fast compute)
            pre = prefetcher.get() if prefetcher else None
            # Immediately arm next step — runs concurrently with all compute below
            next_idx = step_in_epoch + 1
            if prefetcher and next_idx < total_steps_in_epoch:
                prefetcher.submit(*epoch_batches[next_idx], step=current_rollout_step + 1)

            # Subdivide this batch into micro-batches for gradient accumulation
            num_prompts = len(prompts)
            micro_batch_size = max(1, num_prompts // grad_accum_steps)

            accum_grads = None
            accum_metrics: List[Dict[str, float]] = []
            valid_micro_batches = 0

            # Process each micro-batch and accumulate gradients
            micro_idx = 0
            for micro_step in range(0, num_prompts, micro_batch_size):
                micro_idx += 1
                end_idx = min(micro_step + micro_batch_size, num_prompts)
                micro_prompts = prompts[micro_step:end_idx]
                micro_answers = true_answers[micro_step:end_idx]

                logger.info(
                        "\nstep %d — micro-batch %d/%d processing prompts...",
                        step_in_epoch, micro_idx, grad_accum_steps
                    )

                # Slice prefetched rollouts for exactly this micro-batch's prompts
                micro_pre = None
                micro_group_sizes = None
                if pre is not None:
                    texts: List[str] = []
                    targets: List[str] = []
                    questions: List[str] = []
                    micro_group_sizes = []
                    for (rollout_texts, true_answer), prompt in zip(pre[micro_step:end_idx], micro_prompts):
                        micro_group_sizes.append(len(rollout_texts))
                        texts.extend(rollout_texts)
                        targets.extend([true_answer] * len(rollout_texts))
                        questions.extend([prompt] * len(rollout_texts))
                    if texts:
                        micro_pre = (texts, targets, questions)
                    else:
                        micro_group_sizes = None

                result = train_step(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    prompts=micro_prompts,
                    true_answers=micro_answers,
                    epoch_idx=epoch,
                    step_in_epoch=step_in_epoch,
                    total_steps_in_epoch=total_steps_in_epoch,
                    ref_model=ref_model,
                    dtype=dtype,
                    device=device,
                    device_stream=mx.default_stream(device),
                    scaler=scaler,
                    prefetched_rollouts=micro_pre,
                    rollout_group_sizes=micro_group_sizes,
                    step=current_rollout_step,
                )
                
                if result is None:
                    logger.info(
                        "step %d — micro-batch %d/%d had no usable rollouts, skipping",
                        step_in_epoch, micro_idx, grad_accum_steps,
                    )
                    continue
                
                micro_metrics, grads = result
                valid_micro_batches += 1
                
                # Scale gradients before accumulating
                accum_grads = grads if accum_grads is None else _add_grads(accum_grads, grads)
                # Materialise accumulated grads to release the lazy addition graph
                mx.eval(accum_grads)
                accum_metrics.append(micro_metrics)
            
            # If no valid micro-batches produced gradients, skip optimizer step
            if accum_grads is None:
                logger.warning("step %d — no micro-batches produced valid rollouts, skipping step", step_in_epoch)
                continue

            if valid_micro_batches > 1:
                accum_grads = _scale_grads(accum_grads, 1.0 / valid_micro_batches)
                mx.eval(accum_grads)
            
            # --- Apply optimizer update after gradient accumulation ---
            max_grad_norm = float(config.get("max_grad_norm") or 0)
            clip_max = max_grad_norm if max_grad_norm > 0 else float("inf")

            step_grads = accum_grads
            found_inf = False
            if scaler is not None and scaler.enabled:
                step_grads = scaler.unscale(step_grads)
                found_inf = scaler.has_inf_nan(step_grads)

            skipped_update = False
            if found_inf:
                skipped_update = True
                grad_norm = float("nan")
                scaler.update(found_inf=True)
                logger.warning("step %d — AMP overflow detected, skipping optimizer update", step_in_epoch)
            else:
                step_grads, grad_norm_t = optim.clip_grad_norm(step_grads, clip_max)
                mx.eval(grad_norm_t)
                grad_norm = float(grad_norm_t)
                optimizer.update(model, step_grads)
                if scaler is not None and scaler.enabled:
                    scaler.update(found_inf=False)
                mx.eval(model.parameters(), optimizer.state)

            global_step += 1
            metrics = {
                k: sum(m[k] for m in accum_metrics) / len(accum_metrics)
                for k in accum_metrics[0]
                if not isinstance(accum_metrics[0][k], dict)
            }
            # -------------------------------------------------------

            # Save checkpoint weights at a configurable interval.
            should_save_checkpoint = (
                (not skipped_update)
                and save_every_steps > 0
                and global_step % save_every_steps == 0
            )
            should_sync_workers = (
                (not skipped_update)
                and sync_steps > 0
                and global_step % sync_steps == 0
            )

            if should_save_checkpoint or should_sync_workers:
                logger.info("[checkpoint] Step %d — saving policy weights ...", global_step)
                try:
                    # Periodic saves overwrite a stable "latest" checkpoint.
                    weights_dir = save_policy_weights(model, checkpoint_dir, "latest", tokenizer=tokenizer, model_cfg=model_cfg)

                    # Periodically reload vLLM workers so rollouts always come
                    # from the most recent policy.
                    if should_sync_workers:
                        logger.info("[weight_sync] Step %d — syncing workers ...", global_step)
                        sync_and_reload_workers(weights_dir, config, model_config)
                        logger.info("[weight_sync] Step %d — workers reloaded.", global_step)
                        # Discard stale prefetched rollouts (generated by old weights) and re-arm
                        if prefetcher and next_idx < total_steps_in_epoch:
                            prefetcher.flush()
                            prefetcher.submit(*epoch_batches[next_idx], step=current_rollout_step + 1)
                            logger.info("[weight_sync] Prefetcher re-armed for step %d.", next_idx)
                except Exception as sync_exc:  # noqa: BLE001
                    if should_sync_workers:
                        logger.error(
                            "[weight_sync] Step %d — sync failed (training continues): %s",
                            global_step, sync_exc,
                        )
                    else:
                        logger.error(
                            "[checkpoint] Step %d — checkpoint save failed (training continues): %s",
                            global_step, sync_exc,
                        )

            _elapsed = int(time.time() - train_start_time)
            _eh, _em, _es = _elapsed // 3600, (_elapsed % 3600) // 60, _elapsed % 60
            epoch_bar.set_postfix(
                epoch=epoch + 1,
                step=global_step,
                loss=f"{metrics['loss']:.4f}",
                grad_norm=f"{grad_norm:.4f}",
                amp_scale=f"{(scaler.get_scale() if scaler else 1.0):.0f}",
                skipped=int(skipped_update),
                elapsed=f"{_eh:02d}:{_em:02d}:{_es:02d}",
            )
            quality_wandb = {
                f"rewards/{k}": metrics.get(k, 0)
                for k in metrics
                if k.startswith("quality_")
            }
            wandb.log(
                {
                    "train/loss":             metrics["loss"],
                    "rewards/reward_mean":      metrics["reward"],
                    "rewards/reward_std":       metrics.get("reward_std", 0),
                    "rewards/reward_min":       metrics.get("reward_min", 0),
                    "rewards/reward_max":       metrics.get("reward_max", 0),
                    **quality_wandb,
                    "rewards/length_penalty_mean": metrics.get("length_penalty_mean", 0),
                    "rewards/length_penalty_min":  metrics.get("length_penalty_min",  0),
                    "rewards/length_penalty_max":  metrics.get("length_penalty_max",  0),
                    "advantage/advantage_mean":   metrics.get("advantage_mean", 0),
                    "advantage/advantage_std":    metrics.get("advantage_std", 0),
                    "rollouts/generation_token_len_min": metrics.get("generation_token_len_min", 0),
                    "rollouts/generation_token_len_mean": metrics.get("generation_token_len_mean", 0),
                    "rollouts/generation_token_len_max":  metrics.get("generation_token_len_max", 0),
                    "kl/ratio_mean":       metrics.get("ratio_mean", 0),
                    "kl/clip_frac":        metrics.get("clip_frac", 0),
                    "kl/kl_mean":          metrics.get("kl_mean", 0),
                    "train/grad_norm":        grad_norm,
                    "train/amp_scale":        scaler.get_scale() if scaler else 1.0,
                    "train/amp_skipped_step": float(skipped_update),
                    "rollouts/num_rollouts":     metrics["num_rollouts"],
                    "train/epoch":            epoch + 1,
                    "train/step":             global_step,
                },
                step=global_step,
            )
            _publish_dashboard_metrics(
                metrics,
                global_step=global_step,
                total_steps=total_epochs * total_steps_in_epoch,
                grad_norm=grad_norm,
                lr=_optimizer_lr(optimizer, config),
                skipped_update=skipped_update,
            )

            if config.get("do_eval", False) and global_step % val_steps == 0:
                val_metrics = validate(
                    model=model,
                    config=config,
                    val_examples=val_examples,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    ref_model=ref_model,
                    dtype=dtype,
                    device=device,
                )
                wandb.log(val_metrics, step=global_step)

        epoch_bar.close()


def main() -> None:
    
    for _log in [_answers_log_path, _debug_dir / "vllm_rollouts.jsonl"]:
        _log.write_text("", encoding="utf-8")
    
    
    setup_logging(force=True)

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    seed = int(grpo_config.get("seed", 42))
    set_global_seed(seed)
    logger.info("Reproducibility seed set to %d", seed)

    # Resolve device from config and set as MLX global default
    device = get_mlx_device(grpo_config)
    mx.set_default_device(device)
    logger.info("Device: %s", grpo_config.get("device", "cpu").upper())

    dtype = get_dtype_from_config(grpo_config)
    logger.info("Using dtype: %s", dtype)

    model, ref_model, tokenizer, model_cfg = load_model(dtype, grpo_config, model_config)
    train_examples, val_examples = build_train_val_examples(
        grpo_config["data"], seed=seed, tokenizer=tokenizer
    )
    _lora_active = apply_lora_if_quantized(model, grpo_config)
    optimizer_name = grpo_config.get("optimizer", "adam").lower()
    amp_enabled = bool(grpo_config.get("amp", False))
    if optimizer_name == "sgd":
        if amp_enabled:
            logger.warning("AMP is not supported with SGD — disabling AMP")
            amp_enabled = False
        scaler = GradScaler(enabled=False)
        optimizer = optim.SGD(learning_rate=float(grpo_config["learning_rate"]))
        logger.info("Optimizer: SGD (lr=%.6f)", float(grpo_config["learning_rate"]))
    else:
        scaler = GradScaler(
            init_scale=float(grpo_config.get("amp_init_scale", 2 ** 15)),
            growth_interval=int(grpo_config.get("amp_growth_interval", 2000)),
            enabled=amp_enabled,
        )
        master_weights_enabled = _lora_active and bool(grpo_config.get("amp_master_weights", amp_enabled))
        optimizer = MasterWeightAdamW(
            model=model,
            learning_rate=float(grpo_config["learning_rate"]),
            enabled=master_weights_enabled,
        )
        logger.info(
            "Optimizer: AdamW (lr=%.6f, amp=%s, master_weights=%s)",
            float(grpo_config["learning_rate"]), amp_enabled, master_weights_enabled,
        )

    wandb.init(
        project=grpo_config.get("wandb", {}).get("project", "smolcluster-grpo"),
        config={
            **grpo_config,
            "seed": seed,
            "backend": "mlx",
            "hf_model_name": model_config["dp"]["hf_model_name"],
        },
    )

    try:
        train(
            model=model,
            config=grpo_config,
            train_examples=train_examples,
            val_examples=val_examples,
            optimizer=optimizer,
            tokenizer=tokenizer,
            ref_model=ref_model,
            dtype=dtype,
            device=device,
            scaler=scaler,
            model_cfg=model_cfg,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()