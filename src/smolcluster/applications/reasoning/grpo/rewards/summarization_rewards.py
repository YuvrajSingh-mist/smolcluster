"""Reward functions for summarization tasks."""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import evaluate
import mlx.core as mx
from rouge_score import rouge_scorer as _rouge_scorer

logger = logging.getLogger(__name__)

# ROUGE is stateless and thread-safe.
_rouge = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# evaluate/BLEU is thread-safe via thread-local instances.
_local = threading.local()

# NLTK wordnet (used by METEOR) is NOT thread-safe — serialize all meteor calls.
_meteor_lock = threading.Lock()
_meteor_metric = evaluate.load("meteor")


def _get_bleu():
    if not hasattr(_local, "bleu"):
        _local.bleu = evaluate.load("bleu")
    return _local.bleu


def calculate_summary_quality(
    predicted: str,
    reference: str,
    use_rouge: bool = True,
    use_meteor: bool = True,
    use_bleu: bool = True,
) -> dict[str, float]:
    """Compute individual summary quality scores.

    Returns a dict with keys ``rouge_l``, ``meteor``, ``bleu`` (only for
    enabled metrics). Values are in [0, 1]. Higher is better.
    """
    if not predicted.strip() or not reference.strip():
        scores = {}
        if use_rouge:
            scores["rouge_l"] = 0.0
        if use_meteor:
            scores["meteor"] = 0.0
        if use_bleu:
            scores["bleu"] = 0.0
        return scores

    scores = {}

    if use_rouge:
        scores["rouge_l"] = float(_rouge.score(reference, predicted)["rougeL"].fmeasure)

    if use_meteor:
        with _meteor_lock:
            scores["meteor"] = float(_meteor_metric.compute(predictions=[predicted], references=[reference])["meteor"])

    if use_bleu:
        scores["bleu"] = float(_get_bleu().compute(predictions=[predicted], references=[[reference]])["bleu"])

    return scores


def calculate_length_reward(
    predicted_answer: str,
    max_length: int,
    tokenizer: Any | None = None,
) -> float:
    """Reward based on proximity to a target length.

    Uses token count when a tokenizer is provided, otherwise character count.

    Returns a value in (-1, 0], where 0 means exactly max_length.
    """
    if tokenizer is not None:
        hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
        length = len(hf_tok.encode(predicted_answer, add_special_tokens=False))

    # length = len(predicted_answer)
    return -((abs(length - max_length)) / max_length)


# ---------------------------------------------------------------------------
# Per-rollout reward computation (threaded) — used by the GRPO training loops
# ---------------------------------------------------------------------------

MAX_LENGTH_OF_SUMMARIZATION = 64


def _compute_single_reward(
    args: tuple[int, str, str, str, Any, bool, bool, bool, bool],
) -> tuple[int, float, dict]:
    idx, question, generated_text, true_answer, tokenizer, use_rouge, use_meteor, use_bleu, use_length_penalty = args
    quality_scores = calculate_summary_quality(
        generated_text, true_answer,
        use_rouge=use_rouge, use_meteor=use_meteor, use_bleu=use_bleu,
    )
    length_penalty = (
        calculate_length_reward(generated_text, MAX_LENGTH_OF_SUMMARIZATION, tokenizer=tokenizer)
        if use_length_penalty
        else 0.0
    )
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


def compute_summarization_rewards(
    rollout_texts: list[str],
    rollout_targets: list[str],
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    max_workers: int | None = None,
    step: int | None = None,
    rollout_questions: list[str] | None = None,
    tokenizer: Any | None = None,
    use_rouge: bool = False,
    use_meteor: bool = False,
    use_bleu: bool = False,
    use_length_penalty: bool = True,
    log_fn: Callable[[dict], None] | None = None,
) -> tuple[mx.array, dict[str, list[float]]]:
    """Returns (reward_tensor [T*C], components) where components has per-rollout
    lists for each enabled quality metric plus length_penalty and total_reward.

    Args:
        log_fn: Optional callback to persist rollout records (e.g. a file-local
                ``_append_answers_log``). Called with ``{"step": step, "rollouts": [...]}`
                if provided.
    """
    questions = rollout_questions if rollout_questions is not None else [""] * len(rollout_texts)
    indexed_args = [
        (i, q, text, target, tokenizer, use_rouge, use_meteor, use_bleu, use_length_penalty)
        for i, (q, text, target) in enumerate(zip(questions, rollout_texts, rollout_targets, strict=False))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_single_reward, indexed_args))

    reward_values: list[float] = []
    log_records: list[dict] = []
    for _idx, total_reward, log_record in results:
        reward_values.append(total_reward)
        log_records.append(log_record)

    if log_fn is not None:
        log_fn({"step": step, "rollouts": log_records})

    quality_keys = sorted({k for r in log_records for k in r if k.startswith("quality_")})
    components: dict[str, list[float]] = {
        **{k: [r[k] for r in log_records] for k in quality_keys},
        "length_penalty": [r["length_penalty"] for r in log_records],
        "total_reward":   [r["total_reward"]   for r in log_records],
    }

    with mx.stream(mx.default_stream(device)):
        return mx.array(reward_values, dtype=dtype), components
