"""Upload LFM 2.5 350M eval rollouts to HF dataset repo and update README."""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "YuvrajSingh9886/reddit-posts-summarization-grpo"
EVAL_ROLLOUTS_ROOT = (
    Path(__file__).parent.parent
    / "src/smolcluster/applications/reasoning/grpo/evaluation/eval-rollouts"
)
LFM_DIR = EVAL_ROLLOUTS_ROOT / "LFM-2.5-350M-bf16"

api = HfApi()


# ---------------------------------------------------------------------------
# 1. Upload all raw rollout files in one commit via upload_folder
# ---------------------------------------------------------------------------
print("Uploading raw LFM eval rollout files (batched)...")

api.upload_folder(
    folder_path=str(LFM_DIR),
    path_in_repo="raw/LFM-2.5-350M-bf16",
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=["**/*.json"],      # skip results_summary.md etc.
    commit_message="Add LFM-2.5-350M-bf16 eval rollout raw files (all 7 runs)",
)

print("Raw files uploaded.")

# ---------------------------------------------------------------------------
# 2. Build parquet for best LFM run (meteor-rouge) and upload
# ---------------------------------------------------------------------------
print("\nBuilding LFM meteor-rouge viewer parquet...")

rollouts_path = LFM_DIR / "grpo-summarization-length-quality-meteor-rouge" / "rollouts.json"
rollouts = json.loads(rollouts_path.read_text(encoding="utf-8"))

rows = []
for rec in rollouts:
    scores = rec.get("geval_scores") or {}
    rows.append(
        {
            "idx": rec["idx"],
            "document": rec["document"],
            "reference": rec["reference"],
            "generated": rec["generated"],
            "faithfulness": scores.get("Faithfulness"),
            "coverage": scores.get("Coverage"),
            "conciseness": scores.get("Conciseness"),
            "clarity": scores.get("Clarity"),
            "composite": rec.get("geval_composite"),
        }
    )

df = pd.DataFrame(rows)
parquet_path = Path("/tmp/lfm_length_quality_meteor_rouge-00000-of-00001.parquet")
df.to_parquet(parquet_path, index=False)

print(f"  Parquet: {len(df)} rows, columns: {list(df.columns)}")

api.upload_file(
    path_or_fileobj=str(parquet_path),
    path_in_repo="data/lfm_length_quality_meteor_rouge-00000-of-00001.parquet",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add LFM 2.5 350M meteor-rouge viewer parquet (200 rows, best-performing run)",
)
print("Parquet uploaded.")

# ---------------------------------------------------------------------------
# 3. Update README
# ---------------------------------------------------------------------------
print("\nUpdating README...")

new_readme = """\
---
pretty_name: GRPO Summarization Eval Rollouts
task_categories:
- summarization
- text-generation
language:
- en
tags:
- grpo
- evaluation
- summarization
- deepeval
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
    - split: qwen_length_quality_meteor_rouge
      path: data/length_quality_meteor_rouge-00000-of-00001.parquet
    - split: lfm_length_quality_meteor_rouge
      path: data/lfm_length_quality_meteor_rouge-00000-of-00001.parquet
viewer: true
---

# GRPO Summarization Eval Rollouts

Evaluation artifacts for **GRPO summarization checkpoints** from [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster), evaluated on the `mlabonne/smoltldr` validation split (200 examples).

Judge: `gpt-5-mini-2025-08-07` via DeepEval G-Eval (5 rounds averaged)

## Dataset Viewer

The viewer table shows the **length + METEOR + ROUGE** run — the best-performing checkpoint for each model. Each row is one of the 200 validation examples with the source document, reference summary, generated summary, and all four G-Eval scores.

- `qwen_length_quality_meteor_rouge` → Qwen2.5-0.5B-Instruct-bf16 best run
- `lfm_length_quality_meteor_rouge` → LFM2.5-350M-bf16 best run

## Results

### Qwen2.5-0.5B-Instruct-bf16

| Run | Reward Components | Faithfulness | Coverage | Conciseness | Clarity | Composite |
|---|---|---|---|---|---|---|
| length-only | length | 0.678 | 0.407 | 0.592 | 0.739 | 2.416 |
| length-quality | length + ROUGE-L | 0.725 | 0.415 | 0.637 | 0.778 | 2.555 |
| length-quality-bleu | length + BLEU | 0.680 | 0.399 | 0.577 | 0.744 | 2.400 |
| length-quality-bleu-rouge | length + BLEU + ROUGE | 0.810 | 0.502 | 0.650 | 0.770 | 2.732 |
| length-quality-meteor | length + METEOR | — | — | — | — | — |
| length-quality-meteor-bleu | length + METEOR + BLEU | 0.792 | 0.468 | 0.648 | 0.756 | 2.664 |
| **length-quality-meteor-rouge** | **length + METEOR + ROUGE** | **0.832** | **0.511** | **0.659** | **0.767** | **2.769** |

### LFM2.5-350M-bf16

| Run | Reward Components | Faithfulness | Coverage | Conciseness | Clarity | Composite |
|---|---|---|---|---|---|---|
| length-only | length | 0.627 | 0.378 | 0.554 | 0.674 | 2.233 |
| length-quality-bleu | length + BLEU | 0.620 | 0.401 | 0.556 | 0.665 | 2.243 |
| length-quality-rouge | length + ROUGE | 0.642 | 0.414 | 0.575 | 0.646 | 2.278 |
| length-quality-meteor | length + METEOR | 0.689 | 0.433 | 0.595 | 0.641 | 2.358 |
| length-quality-bleu-rouge | length + BLEU + ROUGE | 0.696 | 0.443 | 0.606 | 0.643 | 2.387 |
| length-quality-meteor-bleu | length + METEOR + BLEU | 0.696 | 0.451 | 0.595 | 0.634 | 2.377 |
| **length-quality-meteor-rouge** | **length + METEOR + ROUGE** | **0.834** | **0.493** | **0.685** | **0.690** | **2.701** |

Composite = sum of the four metrics (max 4.0).

## Metric Definitions

All metrics are scored 0–1 by the LLM judge:

- **Faithfulness**: summary stays grounded in the source without hallucinations or contradictions
- **Coverage**: summary captures the source's key points without omitting meaning-critical information
- **Conciseness**: summary is substantially shorter than the source without redundancy
- **Clarity**: summary is easy to read, grammatically sound, and understandable on its own

## File Structure

```
data/
  length_quality_meteor_rouge-00000-of-00001.parquet          # Qwen best run viewer (200 rows)
  lfm_length_quality_meteor_rouge-00000-of-00001.parquet      # LFM best run viewer (200 rows)

raw/
  Qwen2.5-0.5b-Instruct-bf16/
    grpo-summarization-length-only/
      rollouts.json    # per-example documents, generations, and per-round judge scores
      summary.json     # aggregate metric means, pass rates, and run metadata
    grpo-summarization-length-quality/
      rollouts.json
      summary.json
    grpo-summarization-length-quality-bleu/
      rollouts.json
      summary.json
      comparison-vs-grpo-summarization-length-only.json   # paired t-test vs baseline
    grpo-summarization-length-quality-bleu-rouge/
      rollouts.json
      summary.json
    grpo-summarization-length-quality-meteor/
      rollouts.json
      comparison_length_only_vs_length_and_meteor_quality_reward.json
    grpo-summarization-length-quality-meteor-bleu/
      rollouts.json
      summary.json
    grpo-summarization-length-quality-meteor-rouge/
      rollouts.json
      summary.json

  LFM-2.5-350M-bf16/
    grpo-summarization-length-only/
      rollouts.json
      summary.json
    grpo-summarization-length-quality-bleu/
      rollouts.json
      summary.json
      comparison_length_only_vs_length_quality_bleu.json
    grpo-summarization-length-quality-rouge/
      rollouts.json
      summary.json
      comparison_length_only_vs_length_quality_rouge.json
    grpo-summarization-length-quality-meteor/
      rollouts.json
      summary.json
      comparison_length_only_vs_length_quality_meteor.json
    grpo-summarization-length-quality-bleu-rouge/
      rollouts.json
      summary.json
      comparison_length_only_vs_length_quality_bleu_rouge.json
    grpo-summarization-length-quality-meteor-bleu/
      rollouts.json
      summary.json
      comparison_length_only_vs_length_quality_meteor_bleu.json
    grpo-summarization-length-quality-meteor-rouge/
      rollouts.json
      summary.json
      comparison_length_only_vs_length_quality_meteor_rouge.json
```

### Viewer parquet columns

| Column | Type | Description |
|---|---|---|
| `idx` | int | Example index in the validation split |
| `document` | string | Source Reddit post |
| `reference` | string | Human reference summary |
| `generated` | string | Model-generated summary |
| `faithfulness` | float | G-Eval Faithfulness score (0–1) |
| `coverage` | float | G-Eval Coverage score (0–1) |
| `conciseness` | float | G-Eval Conciseness score (0–1) |
| `clarity` | float | G-Eval Clarity score (0–1) |
| `composite` | float | Sum of four metrics (max 4.0) |

### Significance test JSON schema

Paired t-test results (candidate minus baseline) per metric:

```json
{
  "baseline_run": "...",
  "candidate_run": "...",
  "alpha": 0.05,
  "test_name": "paired_t_test",
  "results": {
    "Faithfulness": {
      "baseline_mean": ..., "candidate_mean": ..., "mean_delta": ...,
      "p_value_greater": ..., "p_value_two_sided": ...,
      "significant_greater": ..., "significant_two_sided": ...
    }
  }
}
```
"""

api.upload_file(
    path_or_fileobj=new_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add LFM-2.5-350M-bf16 eval results, parquet split, and file structure; update README",
)
print("README updated.")
print("\nDone! https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo")
