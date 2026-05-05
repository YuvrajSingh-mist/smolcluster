"""Batch inference helpers and retry logic used by GRPO evaluation pipelines."""
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
from scipy import stats

from deepeval.metrics import GEval


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def is_rate_limit_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return (
        "rate_limit" in error_text
        or "too many requests" in error_text
        or "429" in error_text
        or "retryerror" in error_text
    )


def backoff_seconds(attempt_index: int, rate_limited: bool) -> float:
    if rate_limited:
        return min(60.0, 5.0 * math.pow(2, attempt_index - 1))
    return min(10.0, float(attempt_index))


def resolve_path(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def build_geval_metrics(
    judge_model: str,
    metric_specs: List[Dict[str, Any]],
) -> List[GEval]:
    return [GEval(model=judge_model, **spec) for spec in metric_specs]


def normalise_metric_name(metric_name: str) -> str:
    return metric_name.removesuffix(" [GEval]")


def serialise_metric_result(metric_result: Any) -> Optional[Dict[str, Any]]:
    raw_metric_name = getattr(metric_result, "name", None) or getattr(metric_result, "metric", None)
    if raw_metric_name is None:
        return None

    metric_name = normalise_metric_name(raw_metric_name)
    return {
        "name": metric_name,
        "display_name": raw_metric_name,
        "score": getattr(metric_result, "score", None),
        "threshold": getattr(metric_result, "threshold", None),
        "success": getattr(metric_result, "success", None),
        "reason": getattr(metric_result, "reason", None),
        "strict_mode": getattr(metric_result, "strict_mode", getattr(metric_result, "strictMode", None)),
        "evaluation_model": getattr(metric_result, "evaluation_model", getattr(metric_result, "evaluationModel", None)),
        "error": getattr(metric_result, "error", None),
        "evaluation_cost": getattr(metric_result, "evaluation_cost", getattr(metric_result, "evaluationCost", None)),
        "verbose_logs": getattr(metric_result, "verbose_logs", getattr(metric_result, "verboseLogs", None)),
    }


def extract_metric_results(test_result: Any) -> List[Any]:
    metric_results = getattr(test_result, "metrics_data", None)
    if metric_results is None:
        metric_results = getattr(test_result, "metrics_metadata", [])
    return metric_results or []


def parse_test_results(
    test_results: List[Any],
    expected_metric_names: List[str],
    expected_num_tests: Optional[int] = None,
) -> Dict[str, Any]:
    metric_scores: Dict[str, List[float]] = {name: [] for name in expected_metric_names}
    parsed_records: List[Dict[str, Any]] = []
    round_errors: List[str] = []
    total_evaluation_cost = 0.0
    any_evaluation_cost = False
    test_passed = 0
    test_failed = 0

    if expected_num_tests is not None and len(test_results) != expected_num_tests:
        round_errors.append(
            f"Expected {expected_num_tests} test results but received {len(test_results)}"
        )

    for idx, test_result in enumerate(test_results):
        test_success = bool(getattr(test_result, "success", False))
        if test_success:
            test_passed += 1
        else:
            test_failed += 1

        scores_for_example: Dict[str, float] = {}
        metric_details: List[Dict[str, Any]] = []
        metric_errors: List[str] = []

        for metric_result in extract_metric_results(test_result):
            metric_payload = serialise_metric_result(metric_result)
            if metric_payload is None:
                continue

            metric_details.append(metric_payload)
            metric_name = metric_payload["name"]
            metric_score = metric_payload["score"]
            metric_error = metric_payload["error"]
            if metric_error:
                metric_errors.append(f"{metric_name}: {metric_error}")

            metric_cost = metric_payload["evaluation_cost"]
            if metric_cost is not None:
                total_evaluation_cost += float(metric_cost)
                any_evaluation_cost = True

            if metric_score is None:
                continue

            if metric_name in metric_scores:
                metric_scores[metric_name].append(float(metric_score))
                scores_for_example[metric_name] = float(metric_score)

        missing_metric_scores = [
            metric_name for metric_name in expected_metric_names if metric_name not in scores_for_example
        ]
        if missing_metric_scores:
            metric_errors.append(
                f"Missing metric scores: {', '.join(missing_metric_scores)}"
            )

        if metric_errors:
            round_errors.append(f"test[{idx}]: {'; '.join(metric_errors)}")

        parsed_records.append(
            {
                "geval_failed": not test_success,
                "geval_errors": metric_errors,
                "geval_metrics": metric_details,
                "geval_scores": scores_for_example,
                "geval_composite": sum(scores_for_example.values()),
                "geval_passed": test_success,
            }
        )

    is_complete = not round_errors and all(
        len(metric_scores[metric_name]) == len(test_results)
        for metric_name in expected_metric_names
    )

    return {
        "records": parsed_records,
        "metric_scores": metric_scores,
        "round_errors": round_errors,
        "is_complete": is_complete,
        "test_passed": test_passed,
        "test_failed": test_failed,
        "evaluation_cost_usd": total_evaluation_cost if any_evaluation_cost else None,
    }


def aggregate_metric_statistics(
    metric_scores: Dict[str, List[float]],
    metric_thresholds: Dict[str, float],
) -> Dict[str, Any]:
    metric_means: Dict[str, Optional[float]] = {}
    metric_pass_rates: Dict[str, Optional[float]] = {}
    composite = 0.0

    for metric_name, scores in metric_scores.items():
        if scores:
            mean = sum(scores) / len(scores)
            metric_means[metric_name] = mean
            threshold = metric_thresholds[metric_name]
            metric_pass_rates[metric_name] = sum(1 for score in scores if score >= threshold) / len(scores)
            composite += mean
        else:
            metric_means[metric_name] = None
            metric_pass_rates[metric_name] = None

    return {
        "metric_means": metric_means,
        "metric_pass_rates": metric_pass_rates,
        "composite": composite,
    }


def build_significance_report(
    rollout_records: List[Dict[str, Any]],
    metric_thresholds: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    metric_samples: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_thresholds}
    composite_samples: List[float] = []

    for record in rollout_records:
        scores = record.get("geval_scores", {})
        for metric_name in metric_thresholds:
            score = scores.get(metric_name)
            if score is not None:
                metric_samples[metric_name].append(float(score))

        composite_score = record.get("geval_composite")
        if composite_score is not None:
            composite_samples.append(float(composite_score))

    results: Dict[str, Any] = {}
    for metric_name, threshold in metric_thresholds.items():
        results[metric_name] = _one_sample_significance_test(
            metric_samples[metric_name],
            threshold,
            alpha,
            stats.ttest_1samp,
        )

    composite_threshold = float(sum(metric_thresholds.values()))
    results["Composite"] = _one_sample_significance_test(
        composite_samples,
        composite_threshold,
        alpha,
        stats.ttest_1samp,
    )

    return {
        "test_name": "one_sample_t_test",
        "alternative": "greater",
        "null_hypothesis": "mean score <= threshold",
        "alpha": alpha,
        "sample_unit": "example-level final averaged scores across successful judge rounds",
        "num_examples": len(rollout_records),
        "results": results,
    }


def _one_sample_significance_test(
    samples: List[float],
    threshold: float,
    alpha: float,
    ttest_fn: Any,
) -> Dict[str, Any]:
    sample_array = np.asarray(samples, dtype=float)
    sample_size = int(sample_array.size)
    if sample_size == 0:
        return {
            "sample_size": 0,
            "threshold": threshold,
            "mean": None,
            "std": None,
            "t_statistic": None,
            "p_value": None,
            "significant": None,
        }

    mean = float(sample_array.mean())
    std = float(sample_array.std(ddof=1)) if sample_size > 1 else 0.0
    t_statistic, p_value = ttest_fn(sample_array, popmean=threshold, alternative="greater")
    
    return {
        "sample_size": sample_size,
        "threshold": threshold,
        "mean": mean,
        "std": std,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def run_output_dir(base_dir: Path, run_tag: str) -> Path:
    out_dir = base_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_rollouts(
    records: List[Dict[str, Any]],
    base_dir: Path,
    run_tag: str,
) -> Path:
    out_dir = run_output_dir(base_dir, run_tag)
    out_path = out_dir / "rollouts.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def save_summary(summary: Dict[str, Any], base_dir: Path, run_tag: str) -> Path:
    out_dir = run_output_dir(base_dir, run_tag)
    out_path = out_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def save_significance_report(report: Dict[str, Any], base_dir: Path, run_tag: str) -> Path:
    out_dir = run_output_dir(base_dir, run_tag)
    out_path = out_dir / "significance.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path