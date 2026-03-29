import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are an assistant who is good at summarization. "
    "Produce a coherent and concise summary of the given post as the final answer."
)


def _format_prompt(question: str, tokenizer: Optional[Any]) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback for tokenizers without chat template support
    return f"{SYSTEM_PROMPT}\nUser: {question}\nAssistant: "


def build_train_val_examples(
    data_config: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    """Load a HuggingFace dataset and return pre-formatted (prompt, answer) pairs.

    Prompts are formatted with the model's chat template at load time so each
    training step skips the per-step format call.

    Args:
        data_config: Dict with keys ``dataset_name``, ``subset``, ``train_split``,
                     ``val_split`` (matches the ``data:`` section of config.yaml).
        tokenizer:   Tokenizer with ``apply_chat_template`` (instruction-tuned models).
                     Falls back to a plain system/user string if None.

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
        (_format_prompt(q, tokenizer), a)
        for q, a in zip(train_split["prompt"], train_split["completion"])
    ]
    val_examples = [
        (_format_prompt(q, tokenizer), a)
        for q, a in zip(val_split["prompt"], val_split["completion"])
    ]
    return train_examples, val_examples