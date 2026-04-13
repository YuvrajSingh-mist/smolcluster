#!/usr/bin/env python3
"""
G-Eval pipeline for GRPO summarization checkpoints.

Loads the latest checkpoint, generates summaries on the smoltldr test split,
then scores each summary with four independent G-Eval metrics:
  Faithfulness · Coverage · Conciseness · Clarity

Each metric is scored 0–1 (deepeval normalises internally), for a max of 4.0.

Usage:
    python evaluate_summarization.py
    python evaluate_summarization.py --checkpoint-dir checkpoints/grpo-summarization-length-quality/latest
    python evaluate_summarization.py --num-examples 20 --judge-model gpt-4o-mini
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import yaml
from datasets import load_dataset
from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load
from tqdm import tqdm

# Allow running as a standalone script from any directory.
_script_dir = Path(__file__).parent
_smolcluster_root = _script_dir.parents[3]
_project_root = _smolcluster_root.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from smolcluster.applications.reasoning.grpo.data.summarization import PROMPT
from smolcluster.applications.reasoning.grpo.utils.evaluation_utils import (
    aggregate_metric_statistics,
    backoff_seconds,
    batch_items,
    build_significance_report,
    build_geval_metrics,
    is_rate_limit_error,
    parse_test_results,
    resolve_path,
    save_rollouts,
    save_significance_report,
    save_summary,
)
from smolcluster.utils.logging_utils import setup_logging

setup_logging(force=True)
logger = logging.getLogger(__name__)

_EVAL_ROLLOUTS_DIR = _script_dir / "eval-rollouts"

_config_path = _smolcluster_root / "configs" / "inference" / "reasoning" / "grpo" / "config.yaml"
with _config_path.open() as _f:
    _GRPO_CONFIG: Dict[str, Any] = yaml.safe_load(_f)
_DATA_CONFIG: Dict[str, Any] = _GRPO_CONFIG["data"]

# ---------------------------------------------------------------------------
# G-Eval metric specs  (each independently scored 0–1)
# ---------------------------------------------------------------------------

_METRIC_SPECS = [
    {
        "name": "Faithfulness",
        "evaluation_steps": [
            "Extract every factual claim made in the summary (actual output).",
            "For each claim, verify it is explicitly supported by the source document (input).",
            "Penalise any claim that introduces information absent from the source or contradicts it.",
            "A score of 1.0 means every claim is grounded in the source; 0.0 means the summary is entirely hallucinated.",
        ],
        "evaluation_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    },
    {
        "name": "Coverage",
        "evaluation_steps": [
            "Identify the key points, main arguments, and core information in the source document (input).",
            "Check whether each key point is reflected in the summary (actual output).",
            "Penalise omission of important details that substantially change the meaning.",
            "Minor supporting details may be omitted without penalty.",
            "A score of 1.0 means all key points are covered; 0.0 means none are.",
        ],
        "evaluation_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    },
    {
        "name": "Conciseness",
        "evaluation_steps": [
            "Evaluate whether the summary (actual output) is substantially shorter than the source (input).",
            "Penalise redundancy: repeated information, filler phrases, or restating the same point.",
            "Penalise verbosity: including excessive detail that could safely be omitted.",
            "A score of 1.0 means the summary is optimally brief; 0.0 means it is as long or redundant as the source.",
        ],
        "evaluation_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    },
    {
        "name": "Clarity",
        "evaluation_steps": [
            "Evaluate whether the summary (actual output) uses clear, direct language.",
            "Check that all sentences are grammatically well-formed and easy to follow.",
            "Penalise unexplained jargon, ambiguous pronouns, or vague phrasing.",
            "Assess whether a reader unfamiliar with the source would understand the summary on its own.",
            "A score of 1.0 means the summary is perfectly clear; 0.0 means it is incomprehensible.",
        ],
        "evaluation_params": [LLMTestCaseParams.ACTUAL_OUTPUT],
    },
]

_DEFAULT_EVAL_ROUNDS = 5
_DEFAULT_MAX_EVAL_RETRIES_PER_ROUND = 5
_DEFAULT_EVAL_BATCH_SIZE = 10
_DEFAULT_EVAL_MAX_CONCURRENT = 2
_DEFAULT_EVAL_THROTTLE_SECONDS = 0.25
_DEFAULT_INTER_BATCH_SLEEP_SECONDS = 1.0


def _evaluate_round_attempt(
    test_cases: List[LLMTestCase],
    judge_model: str,
    num_eval_rounds: int,
    expected_metric_names: List[str],
    metric_thresholds: Dict[str, float],
    max_eval_retries_per_round: int,
    eval_batch_size: int,
    eval_max_concurrent: int,
    eval_throttle_seconds: float,
    inter_batch_sleep_seconds: float,
    round_index: int,
    attempt_index: int,
) -> Dict[str, Any]:
    combined_records: List[Dict[str, Any]] = []
    combined_metric_scores: Dict[str, List[float]] = {name: [] for name in expected_metric_names}
    combined_test_passed = 0
    combined_test_failed = 0
    combined_evaluation_cost = 0.0
    any_evaluation_cost = False
    total_judge_duration_seconds = 0.0
    batch_failures: List[Dict[str, Any]] = []

    batches = batch_items(test_cases, eval_batch_size)
    async_config = AsyncConfig(
        run_async=eval_max_concurrent > 1,
        throttle_value=eval_throttle_seconds,
        max_concurrent=eval_max_concurrent,
    )
    display_config = DisplayConfig(show_indicator=False, print_results=False)

    for batch_number, batch_test_cases in enumerate(batches, start=1):
        batch_completed = False
        for batch_attempt in range(1, max_eval_retries_per_round + 1):
            rate_limited = False
            batch_metrics = build_geval_metrics(judge_model, _METRIC_SPECS)
            logger.info(
                "Round %d/%d attempt %d/%d batch %d/%d attempt %d/%d (size=%d, max_concurrent=%d, throttle=%.2fs)",
                round_index,
                num_eval_rounds,
                attempt_index,
                max_eval_retries_per_round,
                batch_number,
                len(batches),
                batch_attempt,
                max_eval_retries_per_round,
                len(batch_test_cases),
                eval_max_concurrent,
                eval_throttle_seconds,
            )
            batch_start_time = time.perf_counter()
            try:
                result = deepeval_evaluate(
                    test_cases=batch_test_cases,
                    metrics=batch_metrics,
                    async_config=async_config,
                    display_config=display_config,
                )
                batch_duration_seconds = time.perf_counter() - batch_start_time
                total_judge_duration_seconds += batch_duration_seconds
                parsed_batch = parse_test_results(
                    result.test_results,
                    expected_metric_names,
                    expected_num_tests=len(batch_test_cases),
                )
                if not parsed_batch["is_complete"]:
                    error_message = " | ".join(parsed_batch["round_errors"]) or "Unknown incomplete batch state"
                    batch_failures.append(
                        {
                            "round_index": round_index,
                            "round_attempt_index": attempt_index,
                            "batch_number": batch_number,
                            "batch_attempt_index": batch_attempt,
                            "error": error_message,
                            "judge_duration_seconds": batch_duration_seconds,
                        }
                    )
                    logger.warning(
                        "Round %d attempt %d batch %d attempt %d incomplete: %s",
                        round_index,
                        attempt_index,
                        batch_number,
                        batch_attempt,
                        error_message,
                    )
                else:
                    combined_records.extend(parsed_batch["records"])
                    for metric_name in expected_metric_names:
                        combined_metric_scores[metric_name].extend(parsed_batch["metric_scores"][metric_name])
                    combined_test_passed += parsed_batch["test_passed"]
                    combined_test_failed += parsed_batch["test_failed"]
                    if parsed_batch["evaluation_cost_usd"] is not None:
                        combined_evaluation_cost += parsed_batch["evaluation_cost_usd"]
                        any_evaluation_cost = True
                    batch_completed = True
                    break
            except Exception as exc:
                rate_limited = is_rate_limit_error(exc)
                batch_duration_seconds = time.perf_counter() - batch_start_time
                batch_failures.append(
                    {
                        "round_index": round_index,
                        "round_attempt_index": attempt_index,
                        "batch_number": batch_number,
                        "batch_attempt_index": batch_attempt,
                        "error": str(exc),
                        "judge_duration_seconds": batch_duration_seconds,
                    }
                )
                logger.warning(
                    "Round %d attempt %d batch %d attempt %d raised an exception: %s",
                    round_index,
                    attempt_index,
                    batch_number,
                    batch_attempt,
                    exc,
                )

            sleep_seconds = backoff_seconds(batch_attempt, rate_limited)
            logger.info(
                "Sleeping %.1fs before retrying round %d attempt %d batch %d",
                sleep_seconds,
                round_index,
                attempt_index,
                batch_number,
            )
            time.sleep(sleep_seconds)

        if not batch_completed:
            raise RuntimeError(
                f"Failed round {round_index} attempt {attempt_index} batch {batch_number} after {max_eval_retries_per_round} attempts"
            )

        if inter_batch_sleep_seconds > 0 and batch_number < len(batches):
            logger.info(
                "Sleeping %.1fs before the next batch in round %d attempt %d",
                inter_batch_sleep_seconds,
                round_index,
                attempt_index,
            )
            time.sleep(inter_batch_sleep_seconds)

    round_stats = aggregate_metric_statistics(combined_metric_scores, metric_thresholds)
    return {
        "records": combined_records,
        "metric_scores": combined_metric_scores,
        "test_passed": combined_test_passed,
        "test_failed": combined_test_failed,
        "evaluation_cost_usd": combined_evaluation_cost if any_evaluation_cost else None,
        "judge_duration_seconds": total_judge_duration_seconds,
        "batch_failures": batch_failures,
        **round_stats,
    }

def load_model_from_checkpoint(hf_model_name: str, checkpoint_dir: Path) -> Tuple[Any, Any]:
    """Load base model weights then overlay the GRPO checkpoint."""
    logger.info("Loading base model: %s", hf_model_name)
    model, tokenizer = mlx_load(hf_model_name, tokenizer_config={"trust_remote_code": True})
    mx.eval(model.parameters())

    adapters_path = checkpoint_dir / "adapters" / "adapters.safetensors"
    full_path = checkpoint_dir / "model.safetensors"

    if adapters_path.exists():
        logger.info("Overlaying LoRA adapters from %s", adapters_path)
        flat = mx.load(str(adapters_path))
    elif full_path.exists():
        logger.info("Overlaying full weights from %s", full_path)
        flat = mx.load(str(full_path))
    else:
        raise FileNotFoundError(
            f"No weights found in {checkpoint_dir} "
            "(expected model.safetensors or adapters/adapters.safetensors)"
        )

    model.load_weights(list(flat.items()))
    mx.eval(model.parameters())
    logger.info("Checkpoint loaded.")
    return model, tokenizer


def generate_summary(model: Any, tokenizer: Any, raw_document: str, max_tokens: int) -> str:
    """Format a chat prompt and run greedy generation."""
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": raw_document},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------

def run_eval_pipeline(
    hf_model_name: str,
    checkpoint_dir: Path,
    num_examples: int,
    max_tokens: int,
    judge_model: str,
    num_eval_rounds: int,
    max_eval_retries_per_round: int,
    eval_batch_size: int,
    eval_max_concurrent: int,
    eval_throttle_seconds: float,
    inter_batch_sleep_seconds: float,
) -> None:
    pipeline_start_time = time.perf_counter()
    run_tag = f"{checkpoint_dir.parent.name}__{checkpoint_dir.name}__{time.strftime('%Y%m%d_%H%M%S')}"
    metrics = build_geval_metrics(judge_model, _METRIC_SPECS)
    expected_metric_names = [metric.name for metric in metrics]
    metric_thresholds = {metric.name: metric.threshold for metric in metrics}

    # 1. Load GRPO checkpoint
    model, tokenizer = load_model_from_checkpoint(hf_model_name, checkpoint_dir)

    # 2. Load test split from config
    dataset_name = _DATA_CONFIG["dataset_name"]
    subset = _DATA_CONFIG.get("subset")
    val_split = _DATA_CONFIG["val_split"]
    logger.info("Loading %s / %s split=%s …", dataset_name, subset, val_split)
    ds = load_dataset(dataset_name, subset)[val_split]
    all_examples: List[Tuple[str, str]] = list(zip(ds["prompt"], ds["completion"]))
    examples = all_examples if num_examples == -1 else all_examples[:num_examples]
    logger.info("Evaluating on %d / %d examples.", len(examples), len(all_examples))

    # 3. Generate summaries, build deepeval test cases, and collect rollout records
    test_cases: List[LLMTestCase] = []
    rollout_records: List[Dict[str, Any]] = []
    logger.info("Generating summaries …")
    for idx, (raw_doc, reference) in enumerate(tqdm(examples, desc="Generating")):
        generated = generate_summary(model, tokenizer, raw_doc, max_tokens=max_tokens)
        test_cases.append(
            LLMTestCase(
                input=raw_doc,
                actual_output=generated,
                expected_output=reference,
            )
        )
        rollout_records.append({
            "idx": idx,
            "document": raw_doc,
            "reference": reference,
            "generated": generated,
        })

    # Save raw rollouts before running the (expensive) G-Eval judge
    rollouts_path = save_rollouts(rollout_records, _EVAL_ROLLOUTS_DIR, run_tag)
    logger.info("Saved %d rollout records to %s", len(rollout_records), rollouts_path)

    # 4. Run multiple G-Eval rounds until we complete the requested count.
    successful_rounds: List[Dict[str, Any]] = []
    failed_round_attempts: List[Dict[str, Any]] = []
    logger.info(
        "Running G-Eval with judge: %s (%d successful rounds required, max %d attempt(s) per round, batch_size=%d, max_concurrent=%d, throttle=%.2fs)",
        judge_model,
        num_eval_rounds,
        max_eval_retries_per_round,
        eval_batch_size,
        eval_max_concurrent,
        eval_throttle_seconds,
    )
    for round_index in range(1, num_eval_rounds + 1):
        round_completed = False
        for attempt_index in range(1, max_eval_retries_per_round + 1):
            logger.info(
                "Starting G-Eval round %d/%d attempt %d/%d",
                round_index,
                num_eval_rounds,
                attempt_index,
                max_eval_retries_per_round,
            )
            try:
                round_result = _evaluate_round_attempt(
                    test_cases=test_cases,
                    judge_model=judge_model,
                    num_eval_rounds=num_eval_rounds,
                    expected_metric_names=expected_metric_names,
                    metric_thresholds=metric_thresholds,
                    max_eval_retries_per_round=max_eval_retries_per_round,
                    eval_batch_size=eval_batch_size,
                    eval_max_concurrent=eval_max_concurrent,
                    eval_throttle_seconds=eval_throttle_seconds,
                    inter_batch_sleep_seconds=inter_batch_sleep_seconds,
                    round_index=round_index,
                    attempt_index=attempt_index,
                )
                successful_rounds.append(
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        **round_result,
                    }
                )
                failed_round_attempts.extend(round_result["batch_failures"])
                round_completed = True
                break
            except Exception as exc:
                sleep_seconds = backoff_seconds(attempt_index, is_rate_limit_error(exc))
                failed_round_attempts.append(
                    {
                        "round_index": round_index,
                        "attempt_index": attempt_index,
                        "error": str(exc),
                        "backoff_seconds": sleep_seconds,
                    }
                )
                logger.warning(
                    "G-Eval round %d attempt %d raised an exception: %s",
                    round_index,
                    attempt_index,
                    exc,
                )
                logger.info(
                    "Sleeping %.1fs before retrying round %d",
                    sleep_seconds,
                    round_index,
                )
                time.sleep(sleep_seconds)
        if not round_completed:
            raise RuntimeError(
                f"Failed to complete G-Eval round {round_index} after {max_eval_retries_per_round} attempts"
            )

    # 5. Average successful round outputs and attach them to the rollout records.
    for i, rollout_record in enumerate(rollout_records):
        round_records = [round_result["records"][i] for round_result in successful_rounds]
        average_scores: Dict[str, float] = {}
        average_metrics: List[Dict[str, Any]] = []
        round_successes = [record["geval_passed"] for record in round_records]
        round_errors = [
            error
            for record in round_records
            for error in record["geval_errors"]
        ]

        for metric_name in expected_metric_names:
            metric_rounds = []
            round_scores: List[float] = []
            round_metric_successes: List[Optional[bool]] = []
            round_evaluation_models: List[str] = []
            for record in round_records:
                metric_payload = next(
                    (metric for metric in record["geval_metrics"] if metric["name"] == metric_name),
                    None,
                )
                if metric_payload is None:
                    continue
                metric_rounds.append(metric_payload)
                if metric_payload["score"] is not None:
                    round_scores.append(float(metric_payload["score"]))
                round_metric_successes.append(metric_payload.get("success"))
                evaluation_model = metric_payload.get("evaluation_model")
                if evaluation_model:
                    round_evaluation_models.append(evaluation_model)

            if round_scores:
                average_score = sum(round_scores) / len(round_scores)
                average_scores[metric_name] = average_score
            else:
                average_score = None

            threshold = metric_thresholds[metric_name]
            average_metrics.append(
                {
                    "name": metric_name,
                    "display_name": metric_rounds[0]["display_name"] if metric_rounds else metric_name,
                    "score": average_score,
                    "threshold": threshold,
                    "success": average_score >= threshold if average_score is not None else None,
                    "round_scores": round_scores,
                    "round_successes": round_metric_successes,
                    "evaluation_models": list(dict.fromkeys(round_evaluation_models)),
                }
            )

        round_pass_rate = sum(1 for passed in round_successes if passed) / len(round_successes)
        rollout_record["geval_rounds"] = [
            {
                "round_index": round_result["round_index"],
                "attempt_index": round_result["attempt_index"],
                "judge_duration_seconds": round_result["judge_duration_seconds"],
                **record,
            }
            for round_result, record in zip(successful_rounds, round_records)
        ]
        rollout_record["geval_round_pass_rate"] = round_pass_rate
        rollout_record["geval_failed"] = round_pass_rate < 0.5
        rollout_record["geval_errors"] = round_errors
        rollout_record["geval_metrics"] = average_metrics
        rollout_record["geval_scores"] = average_scores
        rollout_record["geval_composite"] = sum(average_scores.values())

    rollouts_path = save_rollouts(rollout_records, _EVAL_ROLLOUTS_DIR, run_tag)
    logger.info("Saved %d rollout records to %s", len(rollout_records), rollouts_path)

    total_tests = len(test_cases)
    mean_test_passed = sum(round_result["test_passed"] for round_result in successful_rounds) / len(successful_rounds)
    mean_test_failed = sum(round_result["test_failed"] for round_result in successful_rounds) / len(successful_rounds)
    mean_pass_rate = sum(
        round_result["test_passed"] / (round_result["test_passed"] + round_result["test_failed"])
        for round_result in successful_rounds
    ) / len(successful_rounds)
    metric_means = {
        metric_name: sum(
            round_result["metric_means"][metric_name] for round_result in successful_rounds
        ) / len(successful_rounds)
        for metric_name in expected_metric_names
    }
    metric_pass_rates = {
        metric_name: sum(
            round_result["metric_pass_rates"][metric_name] for round_result in successful_rounds
        ) / len(successful_rounds)
        for metric_name in expected_metric_names
    }
    composite = sum(round_result["composite"] for round_result in successful_rounds) / len(successful_rounds)
    total_evaluation_cost = sum(
        round_result["evaluation_cost_usd"] or 0.0 for round_result in successful_rounds
    )

    summary = {
        "run_tag": run_tag,
        "checkpoint_dir": str(checkpoint_dir),
        "judge_model": judge_model,
        "requested_eval_rounds": num_eval_rounds,
        "completed_eval_rounds": len(successful_rounds),
        "max_eval_retries_per_round": max_eval_retries_per_round,
        "eval_batch_size": eval_batch_size,
        "eval_max_concurrent": eval_max_concurrent,
        "eval_throttle_seconds": eval_throttle_seconds,
        "inter_batch_sleep_seconds": inter_batch_sleep_seconds,
        "num_examples": len(examples),
        "total_dataset_examples": len(all_examples),
        "test_passed": mean_test_passed,
        "test_failed": mean_test_failed,
        "total_tests": total_tests,
        "pass_rate": mean_pass_rate if total_tests else None,
        "evaluation_cost_usd": total_evaluation_cost,
        "judge_duration_seconds": sum(round_result["judge_duration_seconds"] for round_result in successful_rounds),
        "wall_clock_seconds": time.perf_counter() - pipeline_start_time,
        "metric_means": metric_means,
        "metric_pass_rates": metric_pass_rates,
        "composite": composite,
        "round_summaries": [
            {
                "round_index": round_result["round_index"],
                "attempt_index": round_result["attempt_index"],
                "judge_duration_seconds": round_result["judge_duration_seconds"],
                "test_passed": round_result["test_passed"],
                "test_failed": round_result["test_failed"],
                "pass_rate": round_result["test_passed"] / (round_result["test_passed"] + round_result["test_failed"]),
                "evaluation_cost_usd": round_result["evaluation_cost_usd"],
                "metric_means": round_result["metric_means"],
                "metric_pass_rates": round_result["metric_pass_rates"],
                "composite": round_result["composite"],
            }
            for round_result in successful_rounds
        ],
        "failed_round_attempts": failed_round_attempts,
    }
    summary_path = save_summary(summary, _EVAL_ROLLOUTS_DIR, run_tag)
    logger.info("Saved evaluation summary to %s", summary_path)

    print(f"Averaged across {len(successful_rounds)} successful G-Eval rounds")
    print("\n" + "=" * 58)
    print(f"{'Metric':<18} {'Mean':>8} {'Pass':>8}   (threshold 0.5)")
    print("-" * 58)
    for metric in metrics:
        mean = metric_means[metric.name]
        pass_rate = metric_pass_rates[metric.name]
        if mean is not None and pass_rate is not None:
            print(f"{metric.name:<18} {mean:>8.4f} {pass_rate:>7.1%}")
        else:
            print(f"{metric.name:<18} {'N/A':>8}")
    print("=" * 58)
    print(f"{'Composite':.<18} {composite:>8.4f} / {len(metrics):.0f}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G-Eval evaluation for summarization GRPO checkpoint")
    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints/grpo-summarization-length-quality-meteor/latest",
        help="Path to checkpoint dir containing model.safetensors (relative to project root or absolute)",
    )
    p.add_argument(
        "--model-name",
        default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        help="Base HuggingFace model ID (must match the checkpoint architecture)",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=-1,
        help="Number of test examples to evaluate (-1 = full dataset, default: -1)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens for summary generation",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-5-mini-2025-08-07",
        help="model used as G-Eval judge (default: gpt-5-mini-2025-08-07)",
    )
    p.add_argument(
        "--num-eval-rounds",
        type=int,
        default=_DEFAULT_EVAL_ROUNDS,
        help=f"Number of successful G-Eval rounds to average (default: {_DEFAULT_EVAL_ROUNDS})",
    )
    p.add_argument(
        "--max-eval-retries-per-round",
        type=int,
        default=_DEFAULT_MAX_EVAL_RETRIES_PER_ROUND,
        help=(
            "Maximum retry attempts for each G-Eval round if an attempt errors or returns incomplete metrics "
            f"(default: {_DEFAULT_MAX_EVAL_RETRIES_PER_ROUND})"
        ),
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=_DEFAULT_EVAL_BATCH_SIZE,
        help=f"Number of test cases to send to DeepEval in each batch (default: {_DEFAULT_EVAL_BATCH_SIZE})",
    )
    p.add_argument(
        "--eval-max-concurrent",
        type=int,
        default=_DEFAULT_EVAL_MAX_CONCURRENT,
        help=f"Maximum concurrent DeepEval tasks within a batch (default: {_DEFAULT_EVAL_MAX_CONCURRENT})",
    )
    p.add_argument(
        "--eval-throttle-seconds",
        type=float,
        default=_DEFAULT_EVAL_THROTTLE_SECONDS,
        help=(
            "Delay inserted by DeepEval between async task launches within a batch "
            f"(default: {_DEFAULT_EVAL_THROTTLE_SECONDS})"
        ),
    )
    p.add_argument(
        "--inter-batch-sleep-seconds",
        type=float,
        default=_DEFAULT_INTER_BATCH_SLEEP_SECONDS,
        help=(
            "Sleep inserted between DeepEval batches to reduce RPM and TPM spikes "
            f"(default: {_DEFAULT_INTER_BATCH_SLEEP_SECONDS})"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    checkpoint_dir = resolve_path(args.checkpoint_dir, _project_root)
    if not checkpoint_dir.is_dir():
        logger.error("Checkpoint directory not found: %s", checkpoint_dir)
        sys.exit(1)

    run_eval_pipeline(
        hf_model_name=args.model_name,
        checkpoint_dir=checkpoint_dir,
        num_examples=args.num_examples,
        max_tokens=args.max_tokens,
        judge_model=args.judge_model,
        num_eval_rounds=args.num_eval_rounds,
        max_eval_retries_per_round=args.max_eval_retries_per_round,
        eval_batch_size=args.eval_batch_size,
        eval_max_concurrent=args.eval_max_concurrent,
        eval_throttle_seconds=args.eval_throttle_seconds,
        inter_batch_sleep_seconds=args.inter_batch_sleep_seconds,
    )