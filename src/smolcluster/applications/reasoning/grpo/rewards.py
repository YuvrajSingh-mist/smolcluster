import re

def calculate_reward(predicted_answer: float, true_answer: float) -> float:
    """
    Calculate the reward based on the predicted answer and the true answer.

    Args:
        predicted_answer: The answer predicted by the model.
        true_answer: The correct answer from the dataset.
    Returns:
        A reward value, which is 1.0 if the predicted answer is correct, and 0.0 otherwise.
    """
    return 1.0 if predicted_answer == true_answer else 0.0


def calculate_formatted_reward(predicted_answer: str) -> float:
    """
    Calculate the reward based on the predicted answer and the true answer, where both are strings.

    Args:
        predicted_answer: The answer predicted by the model, which may contain reasoning steps.
    Returns:
        A reward value, which is 1.0 if the predicted answer is correct, and 0.0 otherwise.
    """
    pattern = r"^<think>(.+?)</think>\s*###\s*(.+)$"
    match = re.match(pattern, predicted_answer.strip(), re.DOTALL)
    if not match:
        return 0.0  
    
    else:
        return 1.0