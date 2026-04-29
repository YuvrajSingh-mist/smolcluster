"""Upload GRPO summarization checkpoints to HuggingFace Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


HF_USER = "YuvrajSingh9886"

CHECKPOINTS = [
    "grpo-summarization-length-only",
    "grpo-summarization-length-quality",
]

PROJECT_ROOT = Path(__file__).parent.parent


def upload_checkpoint(api: HfApi, name: str, readme: str) -> None:
    repo_id = f"{HF_USER}/{name}"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / name / "latest"

    print(f"\nCreating repo {repo_id} ...")
    create_repo(repo_id, repo_type="model", exist_ok=True, token=api.token)

    # Upload README
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    # Upload weights only
    weights = list(ckpt_dir.glob("*.safetensors"))
    if not weights:
        print(f"  WARNING: no safetensors found in {ckpt_dir}")
        return

    for w in weights:
        print(f"  Uploading {w.name} ({w.stat().st_size / 1e6:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(w),
            path_in_repo=w.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {w.name}",
        )

    print(f"  Done → https://huggingface.co/{repo_id}")


def build_readme(name: str) -> str:
    is_quality = "quality" in name

    reward_table = (
        "| Reward | Range | Description |\n"
        "|--------|-------|---------|\n"
        "| `length_penalty` | (-1, 0] | Penalises deviation from 64-token target length. 0 = exact match. |\n"
        "| `quality_reward` | [0, 1] | Composite ROUGE-L + METEOR + BLEU F1 vs reference summary. |\n"
        if is_quality else
        "| Reward | Range | Description |\n"
        "|--------|-------|---------|\n"
        "| `length_penalty` | (-1, 0] | Penalises deviation from 64-token target length. 0 = exact match. |\n"
    )

    reward_summary = (
        "Length penalty **+** composite quality reward (ROUGE-L, METEOR, BLEU averaged)."
        if is_quality else
        "Length penalty only — baseline run with no quality signal."
    )

    return f"""---
license: apache-2.0
base_model: mlx-community/Qwen2.5-0.5B-Instruct-bf16
tags:
  - grpo
  - summarization
  - mlx
  - reinforcement-learning
  - smolcluster
---

# {name}

Fine-tuned from [`mlx-community/Qwen2.5-0.5B-Instruct-bf16`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-bf16)
using **GRPO** (Group Relative Policy Optimisation) on a Reddit summarization task.
Trained with [Smolcluster](https://smolcluster.com) on Apple Silicon via MLX.

**Reward setup:** {reward_summary}

## Models

| Model | Reward Setup |
|-------|-------------|
| [grpo-summarization-length-only](https://huggingface.co/YuvrajSingh9886/grpo-summarization-length-only) | Length penalty only (baseline) |
| [grpo-summarization-length-quality](https://huggingface.co/YuvrajSingh9886/grpo-summarization-length-quality) | Length penalty + ROUGE-L / METEOR / BLEU quality |

## Training

**Algorithm:** GRPO — each step:
1. Samples prompts from `mlabonne/smoltldr` (Reddit posts → summaries).
2. Generates multiple rollouts per prompt via vLLM workers.
3. Scores each rollout with the reward functions below.
4. Normalises rewards within each group → per-rollout advantages.
5. Applies a PPO-style clipped objective, optionally with a KL penalty.
6. Updates the policy on-device via MLX.

**Reward functions:**

{reward_table}

**Dataset:** [`mlabonne/smoltldr`](https://huggingface.co/datasets/mlabonne/smoltldr) — Reddit posts with reference summaries.

**Base model:** `mlx-community/Qwen2.5-0.5B-Instruct-bf16` (Qwen 2.5 0.5B Instruct, bf16 MLX weights)

## Evaluation

Checkpoints are evaluated with a G-Eval LLM-judge pipeline scoring four dimensions per generated summary:
**Faithfulness · Coverage · Conciseness · Clarity**

Eval artifacts are published on HuggingFace:

- Dataset: [YuvrajSingh9886/reddit-posts-summarization-grpo](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo)
  - [`length_only_reward`](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/viewer/length_only_reward) split
  - [`length_and_rouge_quality_reward`](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/viewer/length_and_rouge_quality_reward) split

## Usage (MLX)

```python
from mlx_lm import load, generate

model, tokenizer = load("YuvrajSingh9886/{name}")

messages = [
    {{"role": "system", "content": (
        "You are an assistant who is an expert at summarization. "
        "Summarize the given post, keeping key points and main ideas intact."
    )}},
    {{"role": "user", "content": "<your reddit post here>"}},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128))
```

## About Smolcluster

A distributed deep learning library for training neural networks across heterogeneous hardware.

**Features:**
- Distributed Training: FSDP (ZeRO-optimised), Classic Data Parallelism, Elastic DP, SyncPS, Expert Parallelism, Model Parallelism
- Heterogeneous hardware: Mac minis, Raspberry Pis, MacBooks, Windows machines
- Experiment tracking: W&B integration
- MLX GRPO training on Apple Silicon

**Website:** [smolcluster.com](https://smolcluster.com)

## License

Apache 2.0
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    for name in CHECKPOINTS:
        readme = build_readme(name)
        upload_checkpoint(api, name, readme)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
