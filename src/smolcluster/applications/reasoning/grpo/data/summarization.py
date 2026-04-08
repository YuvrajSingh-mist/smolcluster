import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


PROMPT = (
    "You are an assistant who is an expert at summarization task. "
    "The user gives you a post and you are required to summarize it, keeping the key points and main ideas intact. "
)



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
        logger.error(e)




def build_train_val_examples(
    data_config: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    seed: int = 42,
) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    """Load a HuggingFace dataset and return pre-formatted (prompt, answer) pairs.

    Prompts are formatted at load time so each training step skips the per-step
    format call.

    Args:
        data_config: Dict with keys ``dataset_name``, ``subset``, ``train_split``,
                     ``val_split`` (matches the ``data:`` section of config.yaml).
        seed: Random seed for dataset shuffling reproducibility.

    Returns:
        (train_examples, val_examples) — each a list of (prompt_str, answer_str).
    """
    dataset = load_dataset(
        data_config["dataset_name"],
        data_config.get("subset"),
    )
    train_split = dataset[data_config["train_split"]].shuffle(seed=seed)
    val_split   = dataset[data_config["val_split"]]

    train_examples = [
        (_format_prompt(q, tokenizer), a)
        for q, a in zip(train_split["prompt"], train_split["completion"])
       
    ]
    val_examples = [
        (_format_prompt(q, tokenizer), a)
        for q, a in zip(val_split["prompt"], val_split["completion"])
     
    ]
    return train_examples, val_examples
