"""Reward functions for mathematical reasoning tasks (GSM8K)."""

import math
import re

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
