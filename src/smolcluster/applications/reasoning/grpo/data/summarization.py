import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


PROMPT = (
    "You are an assistant who is good at summarization task. "
    "The user gives you a story and you are required to summarize it "
    "User: {question}. Assistant: "
)



def build_train_val_examples(
    data_config: Dict[str, Any],
) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    """Load a HuggingFace dataset and return pre-formatted (prompt, answer) pairs.

    Prompts are formatted at load time so each training step skips the per-step
    format call.

    Args:
        data_config: Dict with keys ``dataset_name``, ``subset``, ``train_split``,
                     ``val_split`` (matches the ``data:`` section of config.yaml).

    Returns:
        (train_examples, val_examples) — each a list of (prompt_str, answer_str).
    """
    dataset = load_dataset(
        data_config["dataset_name"],
        data_config.get("subset"),
    )
    train_split = dataset[data_config["train_split"]]
    val_split = dataset[data_config["val_split"]]

    train_examples = [
        (PROMPT.format(question=q), None)
        for q in train_split["prompt"]
        # if (ans := extract_answer_from_gsm8k(a)) is not None
    ]
    val_examples = [
        (PROMPT.format(question=q), None)
        for q in val_split["prompt"]
        # if (ans := extract_answer_from_gsm8k(a)) is not None
    ]
    return train_examples, val_examples
