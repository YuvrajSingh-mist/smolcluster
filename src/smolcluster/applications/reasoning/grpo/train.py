import json
import logging
import re
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
from mlx.nn.utils import checkpoint as mlx_grad_checkpoint
from tqdm.auto import tqdm

from smolcluster.applications.reasoning.grpo.data.grpo_data import build_train_val_examples
from smolcluster.applications.reasoning.grpo.amp import GradScaler, MasterWeightAdamW
from smolcluster.applications.reasoning.grpo.rewards import (
    calculate_answer_reward,
    calculate_formatted_reward,
)
from smolcluster.applications.reasoning.grpo.rollouts import build_batched_rollout_texts
from smolcluster.applications.reasoning.grpo.utils import (
    _add_grads,
    _log_mem,
    _scale_grads,
    get_dtype_from_config,
    get_mlx_device,
    iterate_batches,
    load_model,
    tokenize_rollouts,
    parse_numeric_answer,
)
from smolcluster.applications.reasoning.grpo.worker_sync import (
    save_policy_weights,
    sync_and_reload_workers,
)

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


def _append_answers_log(record: dict) -> None:
    with _answers_log_lock:
        with _answers_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")

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
    kl = (ref_logprobs - curr_logprobs) - mx.exp(ref_logprobs - curr_logprobs) - 1
    mx.eval(ratio, kl)
    return {
        "ratio_mean": float(mx.mean(ratio).item()),
        "clip_frac": clip_frac,
        "kl_mean": float(mx.mean(kl).item()),
    }


def compute_advantages(rewards: mx.array, dtype: type = mx.float32) -> mx.array:
    rewards_std = mx.std(rewards)
    return (rewards - mx.mean(rewards)) / mx.maximum(rewards_std, mx.array(1e-3, dtype=dtype))



def compute_grpo_loss(
    curr_logprobs: mx.array,
    advantages: mx.array,
    config: Dict[str, Any],
    ref_logprobs: Optional[mx.array] = None,
) -> mx.array:
    if ref_logprobs is not None and config.get("use_kl", True):
        logprobs_ratio = mx.exp(curr_logprobs - ref_logprobs)
        clipped_ratio = mx.clip(
            logprobs_ratio,
            1 - config["clip_ratio"],
            1 + config["clip_ratio"],
        )
        kl = (ref_logprobs - curr_logprobs) - mx.exp(ref_logprobs - curr_logprobs) - 1
        return -mx.mean(
            mx.minimum(logprobs_ratio * advantages, clipped_ratio * advantages)
            - config["kl_beta"] * kl
        )
    # Plain policy gradient (no KL penalty, no ratio clipping)
    return -mx.mean(advantages * curr_logprobs)


def compute_logprobs(
    model: Any,
    input_ids: mx.array,
    attention_mask: mx.array,
    dtype: type = mx.float32,
    use_checkpoint: bool = False,
) -> mx.array:
    
    
    shift_mask = attention_mask[:, 1:]
    def _forward(ids: mx.array) -> mx.array:
        # All [B, T, vocab_size] tensors live and die inside here.
        # Output is [B, T] so the backward graph stores nothing large.
        logits = model(ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        log_probs = nn.log_softmax(shift_logits, axis=-1)
        return mx.take_along_axis(log_probs, shift_labels[..., None], axis=-1).squeeze(-1)

    if use_checkpoint:
        token_logprobs = mlx_grad_checkpoint(model, _forward)(input_ids)
    else:
        token_logprobs = _forward(input_ids)
    
    valid_mask = mx.isfinite(token_logprobs)
    non_nan_logprobs = mx.where(valid_mask, token_logprobs, 0.0)
    
    non_mask_logprobs = mx.where(shift_mask, non_nan_logprobs, 0.0)
    counts = mx.sum(shift_mask, axis=1)
    
    return mx.sum(non_mask_logprobs, axis=1) / counts


def _compute_single_reward(
    args: Tuple[int, str, str],
) -> Tuple[int, float, dict]:
    idx, generated_text, target_answer = args
    predicted_numeric_answer = parse_numeric_answer(generated_text)
    answer_reward = calculate_answer_reward(predicted_numeric_answer, target_answer)
    format_reward = calculate_formatted_reward(generated_text)
    total_reward = float(answer_reward + format_reward)
    log_record = {
        "rollout_idx":      idx,
        "true_answer":      target_answer,
        "extracted_answer": str(predicted_numeric_answer),
        "answer_reward":    float(answer_reward),
        "format_reward":    float(format_reward),
        "total_reward":     total_reward,
        "generated_text":   generated_text,
    }
    return idx, total_reward, log_record


def compute_rewards(
    rollout_texts: List[str],
    rollout_targets: List[str],
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    max_workers: Optional[int] = None,
) -> mx.array:
    indexed_args = [
        (i, text, target)
        for i, (text, target) in enumerate(zip(rollout_texts, rollout_targets))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map() preserves submission order → results[i] == rollout_texts[i]
        results = list(executor.map(_compute_single_reward, indexed_args))
    # All threads joined here (shutdown(wait=True) called by context manager)

    reward_values: List[float] = []
    log_records: List[dict] = []
    for _idx, total_reward, log_record in results:
        reward_values.append(total_reward)
        log_records.append(log_record)

    _append_answers_log({"rollouts": log_records})

    # Move reward tensor to the configured device
    with mx.stream(mx.default_stream(device)):
        return mx.array(reward_values, dtype=dtype)


def evaluate_batch(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    prompts: List[str],
    true_answers: List[str],
    ref_model: Optional[Any] = None,
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
) -> Optional[Dict[str, float]]:
    rollout_texts, rollout_targets = build_batched_rollout_texts(prompts, true_answers, config)
    if not rollout_texts:
        return None

    input_ids, attention_mask = tokenize_rollouts(tokenizer, rollout_texts, config["max_tokens"], device=device)
    curr_logprobs = compute_logprobs(model, input_ids, attention_mask, dtype=dtype)
    rewards = compute_rewards(rollout_texts, rollout_targets, dtype=dtype, device=device, max_workers=config.get("reward_workers"))
    advantages = compute_advantages(rewards, dtype=dtype)

    ref_logprobs: Optional[mx.array] = None
    if ref_model is not None and config.get("use_kl", True):
        ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask, dtype=dtype)

    loss = compute_grpo_loss(curr_logprobs, advantages, config, ref_logprobs=ref_logprobs)
    mx.eval(loss, rewards)

    return {
        "loss": float(loss.item()),
        "reward": float(mx.mean(rewards).item()),
        "num_rollouts": float(len(rollout_texts)),
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
) -> Optional[Tuple[Dict[str, float], Any]]:
    step_tag = f"[train_step epoch={epoch_idx + 1} step={step_in_epoch}/{total_steps_in_epoch}]"

    logger.info("\n%s Generating rollouts for %d prompt(s) ...", step_tag, len(prompts))
    _t0 = time.time()
    rollout_texts, rollout_targets = build_batched_rollout_texts(prompts, true_answers, config)
    logger.info("%s [TIMING] rollout_gen: %.1fs", step_tag, time.time() - _t0)

    if not rollout_texts:
        logger.warning("%s No rollouts generated, skipping step", step_tag)
        return None
    logger.info("%s Got %d rollout(s)", step_tag, len(rollout_texts))

    input_ids, attention_mask = tokenize_rollouts(tokenizer, rollout_texts, config["max_tokens"], device=device)
    logger.info("%s Tokenized input shape: %s", step_tag, list(input_ids.shape))
    _log_mem("train_step: after tokenize_rollouts")

    logger.info("%s Computing rewards ...", step_tag)
    rewards = compute_rewards(rollout_texts, rollout_targets, dtype=dtype, device=device, max_workers=config.get("reward_workers"))
    logger.info(
        "%s Rewards - mean: %.4f  min: %.4f  max: %.4f",
        step_tag,
        float(mx.mean(rewards).item()),
        float(mx.min(rewards).item()),
        float(mx.max(rewards).item()),
    )
    logger.info("%s Computing advantages ...", step_tag)
    advantages = compute_advantages(rewards, dtype=dtype)
    rewards_std_value = float(mx.std(rewards).item())
    if rewards_std_value < 1e-8:
        logger.warning(
            "%s Reward std is ~0 (%.3e). Advantages collapse and grad_norm may be 0.",
            step_tag,
            rewards_std_value,
        )

    # Compute reference logprobs for KL-penalised / ratio-clipped loss.
    ref_logprobs: Optional[mx.array] = None
    if ref_model is not None and config.get("use_kl", True):
        logger.info("%s Computing reference model log-probs ...", step_tag)
        ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask, dtype=dtype)
        mx.eval(ref_logprobs)

    use_ckpt = config.get("grad_checkpoint", False)
    B = input_ids.shape[0]
    chunk_size = int(config.get("grad_chunk_size", B))  # default = whole batch at once

    mx.reset_peak_memory()
    _log_mem("train_step: before grad computation (peak reset)")
    logger.info("%s Computing GRPO loss and gradients (B=%d chunk_size=%d) ...", step_tag, B, chunk_size)

    accum_grads = None
    total_loss = 0.0
    _t_grad = time.time()

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        # Scale so that accumulated sum of chunk means = full-batch mean
        chunk_scale = (end - start) / B

        c_ids  = input_ids[start:end]
        c_mask = attention_mask[start:end]
        c_adv  = advantages[start:end]
        c_ref  = ref_logprobs[start:end] if ref_logprobs is not None else None

        # Capture loop variables by value via default args to avoid closure bugs
        def chunk_loss_fn(m, _ids=c_ids, _mask=c_mask,
                          _adv=c_adv, _ref=c_ref) -> mx.array:
            lp = compute_logprobs(m, _ids, _mask, dtype=dtype, use_checkpoint=use_ckpt)
            chunk_loss = compute_grpo_loss(lp, _adv, config, ref_logprobs=_ref) * chunk_scale
            if scaler is not None:
                return scaler.scale_loss(chunk_loss)
            return chunk_loss

        chunk_loss, chunk_grads = nn.value_and_grad(model, chunk_loss_fn)(model)
        # Force evaluation: flushes the [chunk, T, vocab] tensors from the graph
        with mx.stream(device_stream):
            mx.eval(chunk_loss, chunk_grads)
        _log_mem(f"train_step: chunk [{start}:{end}] after eval")

        chunk_loss_value = float(chunk_loss.item())
        if scaler is not None and scaler.enabled:
            chunk_loss_value /= scaler.get_scale()
        total_loss += chunk_loss_value
        if accum_grads is None:
            accum_grads = chunk_grads
        else:
            accum_grads = _add_grads(accum_grads, chunk_grads)

    logger.info("%s Loss: %.6f  [TIMING] grad_loop: %.1fs", step_tag, total_loss, time.time() - _t_grad)

    # ---- extra diagnostics (no grad needed) --------------------------------
    lengths = mx.sum(attention_mask, axis=1).astype(dtype)
    metrics_extra: Dict[str, float] = {
        "reward_std":       float(mx.std(rewards).item()),
        "reward_min":       float(mx.min(rewards).item()),
        "reward_max":       float(mx.max(rewards).item()),
        "advantage_mean":   float(mx.mean(advantages).item()),
        "advantage_std":    float(mx.std(advantages).item()),
        "rollout_len_mean": float(mx.mean(lengths).item()),
        "rollout_len_max":  float(mx.max(lengths).item()),
    }
    if ref_logprobs is not None and config.get("use_kl", True):
        curr_lps = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            lp = compute_logprobs(model, input_ids[start:end], attention_mask[start:end], dtype=dtype)
            mx.eval(lp)
            curr_lps.append(lp)
        metrics_extra.update(compute_ratio_stats(
            mx.concatenate(curr_lps, axis=0), ref_logprobs, config, dtype
        ))
    # -------------------------------------------------------------------------

    return {
        "loss": total_loss,
        "reward": float(mx.mean(rewards).item()),
        "num_rollouts": float(len(rollout_texts)),
        **metrics_extra,
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
) -> None:
    model.train()
  
    global_step = 0
    total_epochs = config["num_epochs"]
    val_steps = config["val_steps"]
    base_seed = int(config.get("seed", 42))
    save_steps = int(config.get("weight_sync", {}).get("save_steps", 0))
    checkpoint_dir = str(config.get("weight_sync", {}).get("checkpoint_dir", "checkpoints/grpo"))

    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    logger.info("Starting MLX GRPO training for %s epochs (grad_accum_steps=%d)", total_epochs, grad_accum_steps)

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

        for step_in_epoch, (prompts, true_answers) in enumerate(epoch_bar, start=0):
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
            metrics = {k: sum(m[k] for m in accum_metrics) / len(accum_metrics) for k in accum_metrics[0]}
            # -------------------------------------------------------

            # Periodically save the policy weights and reload vLLM workers so
            # rollouts always come from the most recent policy.
            if (not skipped_update) and save_steps > 0 and global_step % save_steps == 0:
                logger.info("[weight_sync] Step %d — saving and syncing weights ...", global_step)
                try:
                    weights_dir = save_policy_weights(model, checkpoint_dir, global_step)
                    sync_and_reload_workers(weights_dir, config, model_config)
                    logger.info("[weight_sync] Step %d — workers reloaded.", global_step)
                except Exception as sync_exc:  # noqa: BLE001
                    logger.error(
                        "[weight_sync] Step %d — sync failed (training continues): %s",
                        global_step, sync_exc,
                    )

            epoch_bar.set_postfix(
                epoch=epoch + 1,
                step=global_step,
                loss=f"{metrics['loss']:.4f}",
                grad_norm=f"{grad_norm:.4f}",
                amp_scale=f"{(scaler.get_scale() if scaler else 1.0):.0f}",
                skipped=int(skipped_update),
            )
            wandb.log(
                {
                    "train/loss":             metrics["loss"],
                    "train/reward_mean":      metrics["reward"],
                    "train/reward_std":       metrics.get("reward_std", 0),
                    "train/reward_min":       metrics.get("reward_min", 0),
                    "train/reward_max":       metrics.get("reward_max", 0),
                    "train/advantage_mean":   metrics.get("advantage_mean", 0),
                    "train/advantage_std":    metrics.get("advantage_std", 0),
                    "train/rollout_len_mean": metrics.get("rollout_len_mean", 0),
                    "train/rollout_len_max":  metrics.get("rollout_len_max", 0),
                    "train/ratio_mean":       metrics.get("ratio_mean", 0),
                    "train/clip_frac":        metrics.get("clip_frac", 0),
                    "train/kl_mean":          metrics.get("kl_mean", 0),
                    "train/grad_norm":        grad_norm,
                    "train/amp_scale":        scaler.get_scale() if scaler else 1.0,
                    "train/amp_skipped_step": float(skipped_update),
                    "train/num_rollouts":     metrics["num_rollouts"],
                    "train/epoch":            epoch + 1,
                    "train/step":             global_step,
                },
                step=global_step,
            )

            if global_step % val_steps == 0:
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
        
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    seed = int(grpo_config.get("seed", 42))
    np.random.seed(seed)
    mx.random.seed(seed)

    # Resolve device from config and set as MLX global default
    device = get_mlx_device(grpo_config)
    mx.set_default_device(device)
    logger.info("Device: %s", grpo_config.get("device", "cpu").upper())

    dtype = get_dtype_from_config(grpo_config)
    logger.info("Using dtype: %s", dtype)

    train_examples, val_examples = build_train_val_examples(grpo_config["data"])
    model, ref_model, tokenizer = load_model(dtype, grpo_config, model_config)
    amp_enabled = bool(grpo_config.get("amp", False))
    scaler = GradScaler(
        init_scale=float(grpo_config.get("amp_init_scale", 2 ** 15)),
        growth_interval=int(grpo_config.get("amp_growth_interval", 2000)),
        enabled=amp_enabled,
    )
    optimizer = MasterWeightAdamW(
        model=model,
        learning_rate=float(grpo_config["learning_rate"]),
        enabled=bool(grpo_config.get("amp_master_weights", amp_enabled)),
    )

    wandb.init(
        project=grpo_config.get("wandb_project", "smolcluster-grpo"),
        config={
            **grpo_config,
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
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
