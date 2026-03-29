import re
import math
from rouge_score import rouge_scorer as _rouge_scorer

_scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

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


def calculate_formatted_reward(predicted_answer: str) -> float:
    """
    Calculate the reward based on the predicted answer and the true answer, where both are strings.

    Args:
        predicted_answer: The answer predicted by the model, which may contain reasoning steps.
    Returns:
        A reward value, which is 1.0 if the predicted answer is correct, and 0.0 otherwise.
    """
    # has_think = re.search(r"<think>.*?</think>", predicted_answer, re.DOTALL | re.IGNORECASE)
    has_answer = re.search(r"<answer>\s*[-+]?\d*\.?\d+\s*</answer>", predicted_answer, re.DOTALL | re.IGNORECASE)
    # return 1.0 if (has_think and has_answer) else 0.0
    return 1.0 if has_answer else 0.0



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
