import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.metrics import GEval


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