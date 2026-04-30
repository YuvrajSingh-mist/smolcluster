#!/usr/bin/env python3
"""
Prepare GSM8K data for SFT with mlx-lm.

Produces three files under the data/ directory expected by mlx_lm lora:
  data/train.jsonl
  data/valid.jsonl
  data/test.jsonl   (uses GSM8K test split for both valid and test)

Each record uses the "prompt" / "completion" format accepted by mlx_lm
CompletionsDataset (--mask-prompt is supported this way).

The completion wraps the chain-of-thought inside <think>...</think> and
the final numeric answer inside <answer>...</answer>, matching the format
the GRPO reward model targets.

Usage:
    python prepare_data.py                     # write to ./data/
    python prepare_data.py --val-size 200      # use 200 examples for validation
    python prepare_data.py --out-dir my/path   # write to a custom directory
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_smolcluster_root = _here.parents[4]
sys.path.insert(0, str(_smolcluster_root / "src"))

from smolcluster.applications.reasoning.grpo.data.gsm8k import (
    _format_prompt,
    extract_answer_from_gsm8k,
)

def _build_cot(gsm8k_solution: str, numeric_answer: float) -> str:
    """Wrap the GSM8K solution in <think> tags and the answer in <answer> tags.

    GSM8K solutions look like:
        "She has 5 apples. She gives 2 away. So she has 3.\\n#### 3"

    The raw chain-of-thought (everything before ####) goes into <think>.
    The extracted numeric answer goes into <answer>.
    """
    # Strip the #### line — the think block should just be the reasoning.
    cot = re.sub(r"\s*####.*", "", gsm8k_solution).strip()
    # Normalise the answer: if it's a whole number emit as int string
    if numeric_answer == int(numeric_answer):
        ans_str = str(int(numeric_answer))
    else:
        ans_str = str(numeric_answer)
    return f"<think>\n{cot}\n</think>\n<answer>{ans_str}</answer>"


def _build_examples(
    split: Any,
    tokenizer: Any,
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    records = []
    for question, answer_raw in zip(split["question"], split["answer"]):
        numeric = extract_answer_from_gsm8k(answer_raw)
        if numeric is None:
            continue
        prompt = _format_prompt(question, tokenizer)
        completion = _build_cot(answer_raw, numeric)
        records.append({"prompt": prompt, "completion": completion})
        if max_examples is not None and len(records) >= max_examples:
            break
    return records


def _write_jsonl(records: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples → %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K SFT data for mlx-lm LoRA training")
    parser.add_argument("--model-name", default=None,
                        help="HF model or tokenizer ID. Defaults to dp.hf_model_name from model config.")
    parser.add_argument("--out-dir", default=str(_here / "data"),
                        help="Directory to write train/valid/test.jsonl (default: ./data/)")
    parser.add_argument("--val-size", type=int, default=None,
                        help="Cap on validation examples (default: use full GSM8K test split)")
    parser.add_argument("--train-max", type=int, default=None,
                        help="Cap on training examples (default: all)")
    args = parser.parse_args()

    # Resolve model name from config if not provided
    if args.model_name is None:
        import yaml
        cfg_path = _smolcluster_root / "src" / "smolcluster" / "configs" / "inference" / "model_config_inference.yaml"
        with open(cfg_path) as f:
            model_cfg = yaml.safe_load(f)
        args.model_name = model_cfg["dp"]["hf_model_name"]

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    logger.info("Loading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_split = dataset["train"]
    test_split  = dataset["test"]

    logger.info("Building training set ...")
    train_records = _build_examples(train_split, tokenizer, max_examples=args.train_max)

    logger.info("Building validation set ...")
    val_records = _build_examples(test_split, tokenizer, max_examples=args.val_size)

    logger.info("Building test set ...")
    # Reuse the full test split for final evaluation; if --val-size is set we don't
    # double-count — mlx-lm test is only run when --test flag is passed.
    test_records = _build_examples(test_split, tokenizer)

    out = Path(args.out_dir)
    _write_jsonl(train_records, out / "train.jsonl")
    _write_jsonl(val_records,   out / "valid.jsonl")
    _write_jsonl(test_records,  out / "test.jsonl")

    logger.info("Data preparation complete.")
    logger.info("  train : %d examples", len(train_records))
    logger.info("  valid : %d examples", len(val_records))
    logger.info("  test  : %d examples", len(test_records))
    logger.info("Next: bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh")


if __name__ == "__main__":
    main()
