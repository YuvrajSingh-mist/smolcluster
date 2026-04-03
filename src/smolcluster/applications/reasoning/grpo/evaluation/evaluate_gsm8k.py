#!/usr/bin/env python3
"""
Evaluate GSM8K model performance.

Supports two modes:

    Single model — pass@k metrics through the GRPO vLLM rollout workers:

    python evaluate_gsm8k.py --model-path Qwen/Qwen3-0.6B
    python evaluate_gsm8k.py --model-path checkpoints/grpo/final --num-rollouts 8
        python evaluate_gsm8k.py --model-path checkpoints/grpo/latest --use-vllm --num-rollouts 4

  Checkpoint comparison — before/after accuracy comparison (greedy, single pass):

    python evaluate_gsm8k.py --checkpoint-dir checkpoints/grpo
    python evaluate_gsm8k.py --step0 checkpoints/grpo/step_0 --final checkpoints/grpo/step_42
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_unflatten
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml

# Allow running as a standalone script from any directory.
_script_dir = Path(__file__).parent
_smolcluster_root = _script_dir.parents[3]
_project_root = _smolcluster_root.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from smolcluster.applications.reasoning.grpo.utils import parse_answer
from smolcluster.applications.reasoning.grpo.data.gsm8k import build_train_val_examples
from smolcluster.applications.reasoning.grpo.rewards import (
    calculate_answer_reward,
    calculate_formatted_reward,
    calculate_think_reward,
)
from smolcluster.applications.reasoning.grpo.utils.rollouts import (
    build_rollouts_per_prompt,
    build_vllm_worker_urls,
)
from smolcluster.applications.reasoning.grpo.utils.training_utils import get_mlx_device
from smolcluster.applications.reasoning.grpo.utils.worker_sync import sync_and_reload_workers

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
_answers_log_lock = threading.Lock()


def _eval_rollouts_dir() -> Path:
    return _script_dir / "eval-rollouts"


def _eval_answers_log_path() -> Path:
    return _eval_rollouts_dir() / "rollout_answers.jsonl"


def _append_eval_answers_log(record: Dict[str, Any]) -> None:
    answers_path = _eval_answers_log_path()
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with _answers_log_lock:
        with answers_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")


def _build_answer_log_record(
    rollout_idx: int,
    question: str,
    generated_text: str,
    true_answer: float,
) -> Dict[str, Any]:
    predicted_answer = parse_answer(generated_text)
    answer_reward = calculate_answer_reward(predicted_answer, true_answer)
    think_reward = calculate_think_reward(generated_text)
    formatted_reward = calculate_formatted_reward(generated_text)
    total_reward = float(answer_reward + 0.1 * think_reward + 0.1 * formatted_reward)
    return {
        "rollout_idx": rollout_idx,
        "question": question,
        "predicted_answer": predicted_answer,
        "answer_reward": float(answer_reward),
        "think_reward": float(think_reward),
        "formatted_reward": float(formatted_reward),
        "total_reward": total_reward,
        "generated_text": generated_text,
        "true_answer": true_answer,
    }


def enable_eval_rollout_logging() -> Path:
    """Route rollout debug logs to eval-rollouts/ for evaluation runs."""
    eval_rollouts_dir = _eval_rollouts_dir()
    eval_rollouts_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_rollouts_dir / "vllm_rollouts.jsonl"
    answers_path = _eval_answers_log_path()

    # Start each evaluation run with fresh JSONL logs.
    log_path.write_text("", encoding="utf-8")
    answers_path.write_text("", encoding="utf-8")

    os.environ["SMOLCLUSTER_VLLM_DEBUG_LOG_PATH"] = str(log_path)
    logger.info("Evaluation rollout logs: %s", log_path)
    logger.info("Evaluation answer logs: %s", answers_path)
    return log_path


def load_eval_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the GRPO and model configs used by training and evaluation."""
    grpo_cfg_path = _smolcluster_root / "configs" / "inference" / "reasoning" / "grpo" / "config.yaml"
    model_cfg_path = _smolcluster_root / "configs" / "inference" / "model_config_inference.yaml"

    with open(grpo_cfg_path) as f:
        grpo_config = yaml.safe_load(f)
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)
    return grpo_config, model_config


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else _project_root / path


def is_local_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "model.safetensors").exists()
        or (path / "adapters" / "adapters.safetensors").exists()
    )


def load_prompt_tokenizer(model_name: str) -> Any:
    """Load only the tokenizer used to format prompts consistently with training."""
    logger.info("Loading tokenizer from %s", model_name)
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def maybe_sync_vllm_checkpoint(model_path: str, grpo_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    """If model_path points at a local GRPO checkpoint, sync it to the rollout workers."""
    resolved_model_path = resolve_project_path(model_path)
    if is_local_checkpoint_dir(resolved_model_path):
        logger.info("Syncing checkpoint %s to vLLM workers", resolved_model_path)
        sync_and_reload_workers(resolved_model_path, grpo_config, model_config)
        return model_config["dp"]["hf_model_name"]
    return model_path


def get_split_examples(
    grpo_config: Dict[str, Any],
    split: str,
    tokenizer: Any,
    max_examples: Optional[int] = None,
) -> List[Tuple[str, float]]:
    train_examples, val_examples = build_train_val_examples(grpo_config["data"], tokenizer=tokenizer)
    examples = train_examples if split == "train" else val_examples
    if max_examples is not None:
        examples = examples[:max_examples]
        logger.info("Limiting evaluation to %d examples.", max_examples)
    return examples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def configure_eval_device(device_name: str) -> None:
    """Set the MLX default device used by evaluation."""
    device = get_mlx_device({"device": device_name})
    mx.set_default_device(device)
    logger.info("Evaluation device: %s", device_name.upper())

def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    """Load a model and tokenizer from an HF model ID or local checkpoint path."""
    logger.info("Loading model from %s", model_path)
    model, tokenizer = mlx_load(model_path, tokenizer_config={"trust_remote_code": True})
    mx.eval(model.parameters())
    return model, tokenizer


def load_weights_into_model(model: Any, step_dir: Path) -> None:
    """Overlay weights from a checkpoint directory onto an already-loaded model.

    Supports both full model.safetensors and LoRA adapter checkpoints.
    """
    adapter_path = step_dir / "adapters" / "adapters.safetensors"
    full_path = step_dir / "model.safetensors"

    if adapter_path.exists():
        flat = mx.load(str(adapter_path))
        model.load_weights(list(flat.items()))
        logger.info("  Loaded LoRA adapters from %s", adapter_path)
    elif full_path.exists():
        flat = mx.load(str(full_path))
        model.load_weights(list(flat.items()))
        logger.info("  Loaded full weights from %s", full_path)
    else:
        raise FileNotFoundError(
            f"No weights found in {step_dir} "
            "(expected adapters/adapters.safetensors or model.safetensors)"
        )
    mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95
) -> str:
    """Generate a single completion using mlx_lm."""
    sampler = make_sampler(temp=temperature, top_p=top_p)
    kwargs: Dict[str, Any] = {"max_tokens": max_new_tokens, "verbose": False, "sampler": sampler}

    return mlx_generate(model, tokenizer, prompt=prompt, **kwargs)


# ---------------------------------------------------------------------------
# Answer parsing / matching
# ---------------------------------------------------------------------------
def _answers_match(predicted: Optional[float], true: float) -> bool:
    if predicted is None:
        return False
    return calculate_answer_reward(predicted, true) > 0.0


def _estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimate from n samples with c correct completions."""
    if k <= 0 or num_samples <= 0:
        return 0.0
    k = min(k, num_samples)
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - (math.comb(num_samples - num_correct, k) / math.comb(num_samples, k))


# ---------------------------------------------------------------------------
# Pass@k evaluation (multi-rollout)
# ---------------------------------------------------------------------------




def evaluate_pass_k_vllm(
    model_path: str,
    num_rollouts: int = 4,
    max_tokens: int = 512,
    split: str = "test",
    output_file: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate pass@k using the same prompt builder and vLLM rollout path as training."""
    rollout_log_path = enable_eval_rollout_logging()

    grpo_config, model_config = load_eval_configs()
    if not grpo_config.get("vllm", False):
        raise ValueError("GRPO config has vllm=false; enable it to use --use-vllm evaluation.")

    tokenizer_source = model_config["dp"]["hf_model_name"]
    if model_path and model_path != tokenizer_source:
        logger.info(
            "Ignoring --model-path=%s for vLLM pass@k; using model_config dp.hf_model_name=%s",
            model_path,
            tokenizer_source,
        )
    tokenizer = load_prompt_tokenizer(tokenizer_source)
    examples = get_split_examples(grpo_config, split, tokenizer, max_examples=max_examples)

    rollout_config = copy.deepcopy(grpo_config)
    rollout_config["num_rollouts"] = num_rollouts
    rollout_config["max_output_tokens"] = max_tokens

    worker_count = len(build_vllm_worker_urls(grpo_config))
    requested_total_rollouts = worker_count * num_rollouts
    logger.info(
        "Using training-style vLLM rollouts: %d worker(s) x %d rollout(s) per worker = %d completion(s) per prompt",
        worker_count,
        num_rollouts,
        requested_total_rollouts,
    )

    detailed: List[Dict[str, Any]] = []
    rollout_counts: List[int] = []
    rollout_correct_counts: List[int] = []

    for i, (prompt, true_answer) in enumerate(
        tqdm(examples, total=len(examples), desc="Generating vLLM rollouts")
    ):
        prompt_rollouts = build_rollouts_per_prompt(
            [prompt],
            [true_answer],
            rollout_config,
            step=None,
        )[0][0]

        if not prompt_rollouts:
            logger.warning("Question %d returned no usable rollouts; inserting a failing placeholder.", i)
            prompt_rollouts = [""]

        correct_count = 0
        for text in prompt_rollouts:
            predicted = parse_answer(text)
            if _answers_match(predicted, float(true_answer)):
                correct_count += 1

        answer_records = [
            _build_answer_log_record(j, prompt, text, float(true_answer))
            for j, text in enumerate(prompt_rollouts)
        ]
        _append_eval_answers_log({"step": None, "rollouts": answer_records})

        rollout_counts.append(len(prompt_rollouts))
        rollout_correct_counts.append(correct_count)
        detailed.append(
            {
                "question_idx": i,
                "question": prompt,
                "true_answer": float(true_answer),
                "rollouts": prompt_rollouts,
                "num_rollouts": len(prompt_rollouts),
                "num_correct": correct_count,
            }
        )

    effective_total_rollouts = min(rollout_counts) if rollout_counts else 1
    k_values = [1] if effective_total_rollouts <= 1 else [1, effective_total_rollouts]
    pass_at_k_results: Dict[str, float] = {}
    for k in k_values:
        per_question = [
            _estimate_pass_at_k(n, c, k)
            for n, c in zip(rollout_counts, rollout_correct_counts)
        ]
        pass_at_k_results[f"pass@{k}"] = float(sum(per_question) / len(per_question)) if per_question else 0.0

    metrics = {
        **pass_at_k_results,
        "backend": "vllm",
        "num_questions": len(examples),
        "num_workers": worker_count,
        "num_rollouts_per_worker": num_rollouts,
        "requested_total_rollouts": requested_total_rollouts,
        "effective_total_rollouts_min": effective_total_rollouts,
        "effective_total_rollouts_avg": (sum(rollout_counts) / len(rollout_counts)) if rollout_counts else 0.0,
        "rollout_log_path": str(rollout_log_path),
        "answers_log_path": str(_eval_answers_log_path()),
        "timestamp": time.time(),
    }

    print(f"\n{'='*60}")
    print("  GSM8K Pass@k Results")
    print(f"{'='*60}")
    print("  Backend              : vLLM workers (training rollout path)")
    print(f"  Workers              : {worker_count}")
    print(f"  Rollouts per worker  : {num_rollouts}")
    print(f"  Requested total      : {requested_total_rollouts}")
    print(f"  Effective total min  : {effective_total_rollouts}")
    for key in sorted(pass_at_k_results):
        value = pass_at_k_results[key]
        print(f"  {key:<20}: {value:.4f}  ({value*100:.2f}%)")
    print(f"  Questions evaluated : {len(examples)}")
    print(f"{'='*60}\n")

    if output_file:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"metrics": metrics, "detailed_results": detailed}, f, indent=2)
        logger.info("Results saved to %s", out)

    return metrics


# ---------------------------------------------------------------------------
# Accuracy evaluation (single greedy pass, used for comparison mode)
# ---------------------------------------------------------------------------



def evaluate_accuracy_vllm(
    val_examples: List[Tuple[str, float]],
    rollout_config: Dict[str, Any],
    max_new_tokens: int = 512,
    label: str = "",
) -> float:
    """Score exact-match accuracy using vLLM worker rollouts only."""
    exact_match = evaluate.load("exact_match")
    rollout_config = copy.deepcopy(rollout_config)
    rollout_config["num_rollouts"] = 1
    rollout_config["max_output_tokens"] = max_new_tokens
    rollout_config["vllm_request_overrides"] = {"temperature": 0.0, "top_p": 1.0}

    predictions: List[str] = []
    references: List[str] = []

    bar = tqdm(val_examples, desc=f"Evaluating {label}", leave=True)
    for prompt, true_answer in bar:
        prompt_rollouts = build_rollouts_per_prompt([prompt], [true_answer], rollout_config, step=None)[0][0]
        generation = next((text for text in prompt_rollouts if text and text.strip()), "")
        _append_eval_answers_log(
            {
                "step": None,
                "rollouts": [_build_answer_log_record(0, prompt, generation, float(true_answer))],
            }
        )
        predicted = parse_answer(generation)
        pred_str = str(int(predicted)) if (predicted is not None and math.isfinite(predicted) and predicted == int(predicted)) else str(predicted)
        ref_str = str(int(true_answer)) if (math.isfinite(true_answer) and true_answer == int(true_answer)) else str(true_answer)
        predictions.append(pred_str)
        references.append(ref_str)
        running_acc = exact_match.compute(predictions=predictions, references=references)["exact_match"]
        bar.set_postfix(acc=f"{running_acc:.3f}")

    result = exact_match.compute(predictions=predictions, references=references)
    acc = result["exact_match"]
    total = len(val_examples)
    correct = round(acc * total)
    logger.info("[%s] Accuracy: %d / %d = %.4f", label, correct, total, acc)
    return acc


# ---------------------------------------------------------------------------
# Checkpoint comparison helpers
# ---------------------------------------------------------------------------

def _find_step0_and_final(checkpoint_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
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
    step0 = next((d for d in step_dirs if _step_num(d) == 0), step_dirs[0])
    latest_dir = checkpoint_dir / "latest"
    final_dir = latest_dir if latest_dir.is_dir() else step_dirs[-1]
    return step0, final_dir


def run_comparison(
    model_path: str,
    step0_path: Optional[str],
    final_path: Optional[str],
    checkpoint_dir: Optional[str],
    max_new_tokens: int,
    max_examples: Optional[int],
) -> None:
    """Compare checkpoints using vLLM worker generation only."""
    grpo_config, model_config = load_eval_configs()
    enable_eval_rollout_logging()

    # Resolve checkpoint dirs
    if checkpoint_dir is not None:
        ckpt_root = Path(checkpoint_dir)
        if not ckpt_root.is_absolute():
            ckpt_root = _project_root / ckpt_root
    else:
        ckpt_root = _project_root / str(
            grpo_config.get("weight_sync", {}).get("checkpoint_dir", "checkpoints/grpo")
        )

    step0_dir = Path(step0_path) if step0_path else None
    final_dir = Path(final_path) if final_path else None
    if step0_dir is None or final_dir is None:
        auto_step0, auto_final = _find_step0_and_final(ckpt_root)
        step0_dir = step0_dir or auto_step0
        final_dir = final_dir or auto_final

    if step0_dir is None or final_dir is None:
        logger.error(
            "Could not locate checkpoints in %s.\n"
            "  Run training first, or pass --step0 / --final explicitly.",
            ckpt_root,
        )
        sys.exit(1)

    logger.info("step_0 checkpoint : %s", step0_dir)
    logger.info("final  checkpoint : %s", final_dir)

    tokenizer_source = model_path or model_config["dp"]["hf_model_name"]
    if is_local_checkpoint_dir(resolve_project_path(tokenizer_source)):
        tokenizer_source = model_config["dp"]["hf_model_name"]
    tokenizer = load_prompt_tokenizer(tokenizer_source)

    _, val_examples = build_train_val_examples(grpo_config["data"], tokenizer=tokenizer)
    if max_examples is not None:
        val_examples = val_examples[:max_examples]
        logger.info("Limiting evaluation to %d examples.", max_examples)

    rollout_config = copy.deepcopy(grpo_config)
    rollout_config["max_output_tokens"] = max_new_tokens

    logger.info("\n=== Evaluating step_0 (pre-training baseline) ===")
    sync_and_reload_workers(step0_dir, grpo_config, model_config)
    acc_before = evaluate_accuracy_vllm(val_examples, rollout_config, max_new_tokens, label="step_0")

    logger.info("\n=== Evaluating final checkpoint (%s) ===", final_dir.name)
    sync_and_reload_workers(final_dir, grpo_config, model_config)
    acc_after = evaluate_accuracy_vllm(val_examples, rollout_config, max_new_tokens, label=final_dir.name)

    delta = acc_after - acc_before
    print(f"\n{'='*60}")
    print("  GSM8K Accuracy Comparison")
    print(f"{'='*60}")
    print(f"  Before training (step_0)        : {acc_before:.4f}  ({acc_before*100:.2f}%)")
    print(f"  After  training ({final_dir.name:12s})  : {acc_after:.4f}  ({acc_after*100:.2f}%)")
    print(f"  Delta                           : {delta:+.4f}  ({delta*100:+.2f}%)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K. Omit --model-path with --step0/--final for comparison mode."
    )
    parser.add_argument("--model-path", default=None,
                        help="HF model ID or path to MLX checkpoint (e.g. Qwen/Qwen3-0.6B or checkpoints/grpo/final)")
    parser.add_argument("--num-rollouts", type=int, default=8,
                        help="Rollouts requested per worker for vLLM-backed evaluation.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Ignored for evaluation; kept for CLI compatibility")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Ignored for evaluation; kept for CLI compatibility")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Deprecated flag kept for compatibility; evaluation uses the vLLM rollout path")
    parser.add_argument("--device", choices=["cpu", "gpu", "metal"], default="gpu",
                        help="MLX device for evaluation (default: gpu)")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-file", default=None,
                        help="Optional JSON file to save detailed results")
    # Comparison mode args
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Checkpoint root for auto-discovery of step_0 and final")
    parser.add_argument("--step0", default=None, help="Explicit path to step_0 checkpoint")
    parser.add_argument("--final", default=None, help="Explicit path to final checkpoint")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit the number of evaluation examples")

    args = parser.parse_args()
    configure_eval_device(args.device)
    if not args.use_vllm:
        logger.info("Evaluation uses the vLLM rollout backend only.")

    comparison_mode = args.checkpoint_dir or args.step0 or args.final

    if comparison_mode:
        run_comparison(
            model_path=args.model_path,
            step0_path=args.step0,
            final_path=args.final,
            checkpoint_dir=args.checkpoint_dir,
            max_new_tokens=args.max_tokens,
            max_examples=args.max_examples,
        )
    else:
        if args.model_path is None:
            parser.error("--model-path is required for pass@k evaluation mode")
        try:
            if args.temperature != 0.9 or args.top_p != 0.95:
                logger.warning(
                    "Ignoring --temperature/--top-p for pass@k; evaluation uses the same training-style vLLM rollout request path."
                )
            evaluate_pass_k_vllm(
                model_path=args.model_path,
                num_rollouts=args.num_rollouts,
                max_tokens=args.max_tokens,
                split=args.split,
                output_file=args.output_file,
                max_examples=args.max_examples,
            )
        except Exception as e:
            logger.error("Evaluation failed: %s", e, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
