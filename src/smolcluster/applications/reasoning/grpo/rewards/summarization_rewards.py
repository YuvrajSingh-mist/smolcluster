"""Reward functions for summarization tasks."""

from rouge_score import rouge_scorer as _rouge_scorer

_scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def calculate_summary_quality(predicted: str, reference: str) -> float:
    """ROUGE-L F1 between a predicted summary and the gold reference.

    Returns float in [0, 1]. Higher is better.
    Uses stemming so minor morphological variants don't get penalised.
    """
    if not predicted.strip() or not reference.strip():
        return 0.0
    scores = _scorer.score(reference, predicted)
    return float(scores["rougeL"].fmeasure)


def calculate_length_reward(predicted_answer: str, max_length: int) -> float:
    """
    Calculate a reward based on the length of the predicted answer.

    Args:
        predicted_answer: The answer predicted by the model.
        max_length: The maximum length
    Returns:
        A reward value between 0.0 and 1.0, where answers closer to the maximum length receive higher rewards.
    """
    return -((abs(len(predicted_answer) - max_length)) / max_length)
