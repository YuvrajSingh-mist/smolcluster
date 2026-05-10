"""Reward functions for mathematical reasoning tasks (GSM8K)."""

import logging
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

logger = logging.getLogger(__name__)

from ..utils import parse_answer


def calculate_answer_reward(predicted_answer: float, true_answer: float) -> float:
    """
    Calculate the reward based on the predicted answer and the true answer.

    Args:
        predicted_answer: The answer predicted by the model.
        true_answer: The correct answer from the dataset.
    Returns:
        A reward value, which is 1.0 if the predicted answer is correct, and 0.0 otherwise.
    """
    if not math.isfinite(predicted_answer):
        return 0.0
    return 1.0 if math.isclose(predicted_answer, float(true_answer), rel_tol=0.0, abs_tol=1e-6) else 0.0


def calculate_think_reward(predicted_answer: str) -> float:
    """Returns 1.0 if the model used <think>...</think> tags with non-empty reasoning."""
    m = re.search(r"<think>(.*?)</think>", predicted_answer, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return 1.0
    return 0.0


def calculate_formatted_reward(predicted_answer: str) -> float:
    """
    Returns 1.0 if the model used BOTH <think> and <answer> tags correctly:
      - <think>...</think> must be present with non-empty content
      - <answer>...</answer> must contain a parseable number
    Only awards credit when the full expected format is satisfied.
    """
    # Require non-empty <think> tag
    has_think = re.search(r"<think>(.*?)</think>", predicted_answer, re.DOTALL | re.IGNORECASE)
    if not (has_think and has_think.group(1).strip()):
        return 0.0

    # Require <answer> tag with a parseable number
    has_answer = re.search(r"<answer>\s*.*?\s*</answer>", predicted_answer, re.DOTALL | re.IGNORECASE)
    if not has_answer:
        return 0.0

    parsed = parse_answer(predicted_answer)
    return 1.0 if math.isfinite(parsed) else 0.0


# ---------------------------------------------------------------------------
# Per-rollout reward computation (threaded) — used by the GRPO training loops
# ---------------------------------------------------------------------------


def _compute_single_reward(
    args: tuple[int, str, str, str],
) -> tuple[int, float, dict]:
    idx, question, generated_text, true_answer = args
    predicted_answer = parse_answer(generated_text)
    answer_reward = calculate_answer_reward(predicted_answer, true_answer)
    think_reward = calculate_think_reward(generated_text)
    formatted_reward = calculate_formatted_reward(generated_text)
    # Tiered reward:
    #   +0.1 for using <think> tags (any reasoning attempt)
    #   +0.1 for using both <think> and <answer> tags correctly with parseable number
    #   +1.0 for correct answer
    # "Full marks" when all three achieved = 1.2
    total_reward = float(answer_reward + 0.1 * think_reward + 0.1 * formatted_reward)
    log_record = {
        "rollout_idx":      idx,
        "question":         question,
        "predicted_answer": predicted_answer,
        "answer_reward":    float(answer_reward),
        "think_reward":     float(think_reward),
        "formatted_reward": float(formatted_reward),
        "total_reward":     total_reward,
        "generated_text":   generated_text,
        "true_answer":      true_answer,
    }
    return idx, total_reward, log_record


def compute_math_rewards(
    rollout_texts: list[str],
    rollout_targets: list[str],
    dtype: type = mx.float32,
    device: mx.Device = mx.cpu,
    max_workers: int | None = None,
    step: int | None = None,
    rollout_questions: list[str] | None = None,
    log_fn: Callable[[dict], None] | None = None,
) -> tuple[mx.array, dict[str, list[float]]]:
    """Returns (reward_tensor [T*C], components) where components has per-rollout
    lists for each reward term (answer_reward, formatted_reward, total_reward).

    Args:
        log_fn: Optional callback to persist rollout records (e.g. a file-local
                ``_append_answers_log``). Called with ``{"step": step, "rollouts": [...]}`
                if provided.
    """
    questions = rollout_questions if rollout_questions is not None else [""] * len(rollout_texts)
    indexed_args = [
        (i, q, text, target)
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

    components: dict[str, list[float]] = {
        "answer_reward":    [r["answer_reward"]    for r in log_records],
        "formatted_reward": [r["formatted_reward"] for r in log_records],
        "total_reward":     [r["total_reward"]     for r in log_records],
    }

    with mx.stream(mx.default_stream(device)):
        return mx.array(reward_values, dtype=dtype), components
