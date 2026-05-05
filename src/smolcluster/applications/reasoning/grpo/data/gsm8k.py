"""GSM8K dataset loader — formats grade-school math problems and answers for GRPO training and evaluation."""
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
import logging


logger = logging.getLogger(__name__)

PROMPT = (
    "You are an Assistant expert at solving math problems. "
    "The assistant first thinks about the reasoning process "
    "to reach the correct answer within '<think>...</think>' tags and then provides the user with the answer. The FINAL answer must STRICTLY be written as "
    "<answer>answer_here</answer> and the thinking process strictly within '<think>...</think>' tags."
)


# SUMMARIZATION_PROMPT = (
#     "You are an assistant who is good at summarization. "
#     "Produce a coherent and concise summary of the given post as the final answer."
# )


def extract_answer_from_gsm8k(gsm8k_answer: str) -> Optional[float]:
    match = re.search(r"### (.*)", gsm8k_answer)
    if not match:
        return None
    try:
        return float(match.group(1).strip().replace(",", ""))
    except ValueError:
        logger.warning("[data] Failed to parse GSM8K answer as float: %.80r", gsm8k_answer)
        return None
    

def _format_prompt(question: str, tokenizer: Optional[Any]) -> str:
    
    try:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": question},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    # Fallback for tokenizers without chat template support
    except Exception as e:
        logger.error("[data] chat-template formatting failed, returning None: %s", e)



def build_train_val_examples(
    data_config: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    seed: int = 42,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Load a HuggingFace dataset and return pre-formatted (prompt, answer) pairs.

    Prompts are formatted at load time so each training step skips the per-step
    format call. Examples whose answer cannot be parsed to float are silently skipped.

    Args:
        data_config: Dict with keys ``dataset_name``, ``subset``, ``train_split``,
                     ``val_split`` (matches the ``data:`` section of config.yaml).
        tokenizer: Optional tokenizer for chat-template formatting.
        seed: Random seed for dataset shuffling reproducibility.

    Returns:
        (train_examples, val_examples) — each a list of (prompt_str, answer_float).
    """
    dataset = load_dataset(
        data_config["dataset_name"],
        data_config.get("subset"),
    )
    train_split = dataset[data_config["train_split"]].shuffle(seed=seed)
    val_split   = dataset[data_config["val_split"]]

    train_examples = [
        (_format_prompt(q, tokenizer), ans)
        for q, a in zip(train_split["question"], train_split["answer"])
        if (ans := extract_answer_from_gsm8k(a)) is not None
    ]
    val_examples = [
        (_format_prompt(q, tokenizer), ans)
        for q, a in zip(val_split["question"], val_split["answer"])
        if (ans := extract_answer_from_gsm8k(a)) is not None
    ]
    return train_examples, val_examples