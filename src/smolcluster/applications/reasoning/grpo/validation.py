"""
GSM8K validation script.

Loads two checkpoints — the initial step_0 weights (pre-training baseline)
and the highest-step final weights — then evaluates exact-match accuracy on
the GSM8K test split for both, printing a side-by-side comparison.

Usage:
    python validation.py
    python validation.py --checkpoint-dir checkpoints/grpo
    python validation.py --checkpoint-dir checkpoints/grpo --max-examples 200
    python validation.py --step0 checkpoints/grpo/step_0 --final checkpoints/grpo/step_42
"""

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import yaml
from mlx_lm import load as mlx_load
from mlx.utils import tree_unflatten
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root and configs
# ---------------------------------------------------------------------------
_module_dir = Path(__file__).parent
_smolcluster_root = _module_dir.parents[2]
_project_root = _smolcluster_root.parent.parent

_grpo_config_path = _smolcluster_root / "configs" / "inference" / "reasoning" / "grpo" / "config.yaml"
_model_config_path = _smolcluster_root / "configs" / "inference" / "model_config_inference.yaml"

with open(_grpo_config_path) as _f:
    _grpo_config = yaml.safe_load(_f)

with open(_model_config_path) as _f:
    _model_config = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_answer_from_generation(text: str) -> Optional[float]:
    """Extract the numeric answer from model output.
    Tries <answer>...</answer> first, then 'final answer' fallback."""
    m = re.findall(r"<answer>\s*([-+]?\d*\.?\d+)\s*</answer>", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass
    m2 = re.findall(r"final\s+answer[^\d\-\+]*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2[-1])
        except ValueError:
            pass
    return None


def _answers_match(predicted: Optional[float], true: float) -> bool:
    if predicted is None or not math.isfinite(predicted):
        return False
    return math.isclose(predicted, true, rel_tol=0.0, abs_tol=1e-6)


def _find_step0_and_final(checkpoint_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (step_0_dir, highest_step_dir) inside checkpoint_dir."""
    if not checkpoint_dir.is_dir():
        return None, None

    step_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if not step_dirs:
        return None, None

    def _step_num(d: Path) -> int:
        try:
            return int(d.name.split("_")[1])
        except (IndexError, ValueError):
            return -1

    step_dirs.sort(key=_step_num)
    step0 = next((d for d in step_dirs if _step_num(d) == 0), None)
    final = step_dirs[-1] if step_dirs else None
    # If only one checkpoint exists return it as both
    if step0 is None:
        step0 = step_dirs[0]
    return step0, final


def _load_weights_into_model(model: Any, step_dir: Path) -> None:
    """Load safetensors weights from a checkpoint directory into an MLX model in-place."""
    # LoRA adapters path
    adapter_path = step_dir / "adapters" / "adapters.safetensors"
    full_path    = step_dir / "model.safetensors"

    if adapter_path.exists():
        flat = mx.load(str(adapter_path))
        params = dict(model.parameters())
        params.update(flat)  # overlay adapter weights
        model.load_weights(list(flat.items()))
        logger.info("  Loaded LoRA adapters from %s", adapter_path)
    elif full_path.exists():
        flat = mx.load(str(full_path))
        model.load_weights(list(flat.items()))
        logger.info("  Loaded full weights from %s", full_path)
    else:
        raise FileNotFoundError(
            f"No weights found in {step_dir} "
            f"(expected adapters/adapters.safetensors or model.safetensors)"
        )
    mx.eval(model.parameters())


def _run_greedy(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 512) -> str:
    """Run greedy decoding via the mlx_lm generate API (no external vLLM needed)."""
    from mlx_lm import generate as mlx_generate
    result = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
    return result


def evaluate_accuracy(
    model: Any,
    tokenizer: Any,
    val_examples: List[Tuple[str, float]],
    max_new_tokens: int = 512,
    label: str = "",
) -> float:
    """Generate answers for every val example and return exact-match accuracy."""
    model.eval()
    correct = 0
    total = len(val_examples)

    bar = tqdm(val_examples, desc=f"Evaluating {label}", leave=True)
    for prompt, true_answer in bar:
        generation = _run_greedy(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        predicted  = _parse_answer_from_generation(generation)
        if _answers_match(predicted, true_answer):
            correct += 1
        bar.set_postfix(acc=f"{correct / max(1, bar.n):.3f}")

    acc = correct / total if total > 0 else 0.0
    logger.info("[%s] Accuracy: %d / %d = %.4f", label, correct, total, acc)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K accuracy before and after GRPO training.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint root (e.g. checkpoints/grpo). "
             "Relative paths are resolved from the project root. "
             "Defaults to the value of weight_sync.checkpoint_dir in config.yaml.",
    )
    parser.add_argument(
        "--step0",
        type=str,
        default=None,
        help="Override: explicit path to the step_0 checkpoint directory.",
    )
    parser.add_argument(
        "--final",
        type=str,
        default=None,
        help="Override: explicit path to the final checkpoint directory.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit evaluation to the first N val examples (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per example.",
    )
    args = parser.parse_args()

    # Resolve checkpoint root
    if args.checkpoint_dir is not None:
        ckpt_root = Path(args.checkpoint_dir)
        if not ckpt_root.is_absolute():
            ckpt_root = _project_root / ckpt_root
    else:
        ckpt_root = _project_root / str(
            _grpo_config.get("weight_sync", {}).get("checkpoint_dir", "checkpoints/grpo")
        )

    # Resolve step_0 and final dirs
    if args.step0 is not None:
        step0_dir = Path(args.step0)
    else:
        step0_dir, _ = _find_step0_and_final(ckpt_root)

    if args.final is not None:
        final_dir = Path(args.final)
    else:
        _, final_dir = _find_step0_and_final(ckpt_root)

    if step0_dir is None or final_dir is None:
        logger.error(
            "Could not locate checkpoints in %s.\n"
            "  Run training first, or pass --step0 / --final explicitly.",
            ckpt_root,
        )
        sys.exit(1)

    logger.info("step_0 checkpoint : %s", step0_dir)
    logger.info("final  checkpoint : %s", final_dir)

    # Load tokenizer config (same as training)
    tokenizer_config = {
        "trust_remote_code": True if _grpo_config.get("trust_remote_code", False) else None
    }
    model_name = _model_config["dp"]["hf_model_name"]

    # Build val examples
    from smolcluster.applications.reasoning.grpo.data.gsm8k import build_train_val_examples
    logger.info("Loading GSM8K val split ...")
    _, val_examples_raw = build_train_val_examples(
        _grpo_config["data"],
        tokenizer=None,  # format with tokenizer below once loaded
    )

    # Load model + tokenizer once (we'll swap weights in-place — no double memory)
    logger.info("Loading base model: %s", model_name)
    model, tokenizer = mlx_load(model_name, tokenizer_config=tokenizer_config)
    mx.eval(model.parameters())

    # Re-build val examples with the real tokenizer (chat template applied)
    _, val_examples = build_train_val_examples(_grpo_config["data"], tokenizer=tokenizer)
    if args.max_examples is not None:
        val_examples = val_examples[: args.max_examples]
        logger.info("Limiting evaluation to %d examples.", args.max_examples)

    max_new_tokens = args.max_new_tokens

    # -----------------------------------------------------------------------
    # Evaluate step_0 (pre-training baseline)
    # -----------------------------------------------------------------------
    logger.info("\n=== Evaluating step_0 (initial / pre-training) ===")
    _load_weights_into_model(model, step0_dir)
    acc_before = evaluate_accuracy(model, tokenizer, val_examples, max_new_tokens, label="step_0")

    # -----------------------------------------------------------------------
    # Evaluate final checkpoint
    # -----------------------------------------------------------------------
    logger.info("\n=== Evaluating final checkpoint (%s) ===", final_dir.name)
    _load_weights_into_model(model, final_dir)
    acc_after = evaluate_accuracy(model, tokenizer, val_examples, max_new_tokens, label=final_dir.name)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GSM8K Accuracy Comparison")
    print("=" * 60)
    print(f"  Before training (step_0)  : {acc_before:.4f}  ({acc_before * 100:.2f}%)")
    print(f"  After  training ({final_dir.name:12s}): {acc_after:.4f}  ({acc_after * 100:.2f}%)")
    delta = acc_after - acc_before
    print(f"  Delta                     : {delta:+.4f}  ({delta * 100:+.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
