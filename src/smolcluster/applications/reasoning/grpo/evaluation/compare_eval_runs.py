#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).parent
_EVAL_ROLLOUTS_DIR = _SCRIPT_DIR / "eval-rollouts"
_METRIC_NAMES = ["Faithfulness", "Coverage", "Conciseness", "Clarity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two saved summarization eval runs with paired tests.")
    parser.add_argument("--baseline-run", required=True, help="Baseline run directory name under eval-rollouts.")
    parser.add_argument("--candidate-run", required=True, help="Candidate run directory name under eval-rollouts.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    return parser.parse_args()


def resolve_run_dir(run_name: str) -> Path:
    run_dir = _EVAL_ROLLOUTS_DIR / run_name
    if not (run_dir / "rollouts.json").exists():
        raise FileNotFoundError(f"No rollouts.json found for run: {run_name}")
    return run_dir


def final_scores_for_record(record: Dict[str, object]) -> Tuple[Dict[str, float], Optional[float]]:
    top_scores = record.get("geval_scores")
    top_composite = record.get("geval_composite")
    if isinstance(top_scores, dict):
        scores = {
            metric_name: float(score)
            for metric_name, score in top_scores.items()
            if score is not None and metric_name in _METRIC_NAMES
        }
        composite = float(top_composite) if top_composite is not None else None
        return scores, composite

    metric_values: Dict[str, List[float]] = {metric_name: [] for metric_name in _METRIC_NAMES}
    composite_values: List[float] = []
    for round_result in record.get("geval_rounds", []):
        if not isinstance(round_result, dict):
            continue
        scores = round_result.get("geval_scores", {})
        if isinstance(scores, dict):
            for metric_name in _METRIC_NAMES:
                value = scores.get(metric_name)
                if value is not None:
                    metric_values[metric_name].append(float(value))

        composite_score = round_result.get("geval_composite")
        if composite_score is not None:
            composite_values.append(float(composite_score))

    averaged_scores = {
        metric_name: float(np.mean(values))
        for metric_name, values in metric_values.items()
        if values
    }
    averaged_composite = float(np.mean(composite_values)) if composite_values else None
    return averaged_scores, averaged_composite


def load_scores_by_idx(run_dir: Path) -> Dict[int, Dict[str, float]]:
    rollouts = json.loads((run_dir / "rollouts.json").read_text(encoding="utf-8"))
    scores_by_idx: Dict[int, Dict[str, float]] = {}
    for record in rollouts:
        idx = record.get("idx")
        if idx is None:
            continue
        scores, composite = final_scores_for_record(record)
        payload = dict(scores)
        if composite is not None:
            payload["Composite"] = composite
        scores_by_idx[int(idx)] = payload
    return scores_by_idx


def paired_test(baseline: List[float], candidate: List[float], alpha: float) -> Dict[str, float]:
    baseline_arr = np.asarray(baseline, dtype=float)
    candidate_arr = np.asarray(candidate, dtype=float)
    delta_arr = candidate_arr - baseline_arr
    sample_size = int(delta_arr.size)
    mean_baseline = float(baseline_arr.mean())
    mean_candidate = float(candidate_arr.mean())
    mean_delta = float(delta_arr.mean())
    std_delta = float(delta_arr.std(ddof=1)) if sample_size > 1 else 0.0

    if sample_size == 0:
        return {
            "sample_size": 0,
            "baseline_mean": None,
            "candidate_mean": None,
            "mean_delta": None,
            "std_delta": None,
            "t_statistic": None,
            "p_value_greater": None,
            "p_value_two_sided": None,
            "significant_greater": None,
            "significant_two_sided": None,
        }

    if sample_size == 1 or np.allclose(delta_arr, delta_arr[0]):
        if mean_delta > 0:
            t_statistic = float("inf")
            p_value_greater = 0.0
            p_value_two_sided = 0.0
        elif mean_delta < 0:
            t_statistic = float("-inf")
            p_value_greater = 1.0
            p_value_two_sided = 0.0
        else:
            t_statistic = 0.0
            p_value_greater = 1.0
            p_value_two_sided = 1.0
    else:
        greater_result = stats.ttest_rel(candidate_arr, baseline_arr, alternative="greater")
        two_sided_result = stats.ttest_rel(candidate_arr, baseline_arr, alternative="two-sided")
        t_statistic = float(greater_result.statistic)
        p_value_greater = float(greater_result.pvalue)
        p_value_two_sided = float(two_sided_result.pvalue)

    return {
        "sample_size": sample_size,
        "baseline_mean": mean_baseline,
        "candidate_mean": mean_candidate,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "t_statistic": t_statistic,
        "p_value_greater": p_value_greater,
        "p_value_two_sided": p_value_two_sided,
        "significant_greater": p_value_greater < alpha,
        "significant_two_sided": p_value_two_sided < alpha,
    }


def main() -> None:
    args = parse_args()
    baseline_dir = resolve_run_dir(args.baseline_run)
    candidate_dir = resolve_run_dir(args.candidate_run)

    baseline_scores = load_scores_by_idx(baseline_dir)
    candidate_scores = load_scores_by_idx(candidate_dir)
    shared_indices = sorted(set(baseline_scores) & set(candidate_scores))
    if not shared_indices:
        raise ValueError("No overlapping example indices between the two runs")

    results: Dict[str, Dict[str, float]] = {}
    for metric_name in [*_METRIC_NAMES, "Composite"]:
        baseline_values: List[float] = []
        candidate_values: List[float] = []
        for idx in shared_indices:
            baseline_value = baseline_scores[idx].get(metric_name)
            candidate_value = candidate_scores[idx].get(metric_name)
            if baseline_value is None or candidate_value is None:
                continue
            baseline_values.append(float(baseline_value))
            candidate_values.append(float(candidate_value))
        results[metric_name] = paired_test(baseline_values, candidate_values, args.alpha)

    report = {
        "baseline_run": baseline_dir.name,
        "candidate_run": candidate_dir.name,
        "alpha": args.alpha,
        "test_name": "paired_t_test",
        "difference_direction": "candidate_minus_baseline",
        "alternative_greater": "candidate mean > baseline mean",
        "num_shared_examples": len(shared_indices),
        "results": results,
    }

    out_path = candidate_dir / f"comparison-vs-{baseline_dir.name}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Baseline: {baseline_dir.name}")
    print(f"Candidate: {candidate_dir.name}")
    print(f"Saved: {out_path}")
    for metric_name, payload in results.items():
        print(
            f"{metric_name}: baseline={payload['baseline_mean']:.6f} candidate={payload['candidate_mean']:.6f} "
            f"delta={payload['mean_delta']:.6f} p_greater={payload['p_value_greater']:.6g} "
            f"p_two_sided={payload['p_value_two_sided']:.6g} significant_greater={payload['significant_greater']}"
        )


if __name__ == "__main__":
    main()