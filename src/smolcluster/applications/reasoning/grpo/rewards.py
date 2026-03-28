import re
import math

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



def calculate_length_reward(predicted_answer: str, max_length: int) -> float:
    """
    Calculate a reward based on the length of the predicted answer.

    Args:
        predicted_answer: The answer predicted by the model.
        max_length: The maximum length 
    Returns:
        A reward value between 0.0 and 1.0, where answers closer to the maximum length receive higher rewards.
    """
    
    if len(predicted_answer) > max_length:

        return -((len(predicted_answer) - max_length) / max_length)
    
    elif len(predicted_answer) < max_length:

        return -((max_length - len(predicted_answer)) / max_length)
    
    return 1.0