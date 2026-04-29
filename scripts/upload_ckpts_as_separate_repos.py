"""
Upload each GRPO checkpoint run as its own HF model repo.

Repo naming:
  LFM2.5-350M-bf16  /  grpo-summarization-X  →  YuvrajSingh9886/LFM2.5-350M-grpo-summarization-X
  Qwen2.5-0.5B-instruct-bf16  /  grpo-summarization-X  →  YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-X
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USER = "YuvrajSingh9886"

MODELS = [
    {
        "ckpt_dir": Path(__file__).parent.parent / "checkpoints" / "LFM2.5-350M-bf16",
        "repo_prefix": "LFM2.5-350M",
        "base_model": "liquid-ai/LFM-2.5-350M",
        "generation_config_source": None,
    },
    {
        "ckpt_dir": Path(__file__).parent.parent / "checkpoints" / "Qwen2.5-0.5B-instruct-bf16",
        "repo_prefix": "Qwen2.5-0.5B",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "generation_config_source": "Qwen/Qwen2.5-0.5B-Instruct",
    },
]

# G-Eval results (composite) for README cards
RESULTS = {
    "LFM2.5-350M": {
        "grpo-summarization-length-only":                 {"composite": 2.233, "faithfulness": 0.627, "coverage": 0.378, "conciseness": 0.554, "clarity": 0.674},
        "grpo-summarization-length-quality-bleu":         {"composite": 2.243, "faithfulness": 0.620, "coverage": 0.401, "conciseness": 0.556, "clarity": 0.665},
        "grpo-summarization-length-quality-rouge":        {"composite": 2.278, "faithfulness": 0.642, "coverage": 0.414, "conciseness": 0.575, "clarity": 0.646},
        "grpo-summarization-length-quality-meteor":       {"composite": 2.358, "faithfulness": 0.689, "coverage": 0.433, "conciseness": 0.595, "clarity": 0.641},
        "grpo-summarization-length-quality-bleu-rouge":   {"composite": 2.387, "faithfulness": 0.696, "coverage": 0.443, "conciseness": 0.606, "clarity": 0.643},
        "grpo-summarization-length-quality-meteor-bleu":  {"composite": 2.377, "faithfulness": 0.696, "coverage": 0.451, "conciseness": 0.595, "clarity": 0.634},
        "grpo-summarization-length-quality-meteor-rouge": {"composite": 2.701, "faithfulness": 0.834, "coverage": 0.493, "conciseness": 0.685, "clarity": 0.690},
    },
    "Qwen2.5-0.5B": {
        "grpo-summarization-length-quality-bleu":         {"composite": 2.400, "faithfulness": 0.680, "coverage": 0.399, "conciseness": 0.577, "clarity": 0.744},
        "grpo-summarization-length-quality-rouge":        {"composite": None,  "faithfulness": None,  "coverage": None,  "conciseness": None,  "clarity": None},
        "grpo-summarization-length-quality-meteor":       {"composite": None,  "faithfulness": None,  "coverage": None,  "conciseness": None,  "clarity": None},
        "grpo-summarization-length-quality-bleu-rouge":   {"composite": 2.732, "faithfulness": 0.810, "coverage": 0.502, "conciseness": 0.650, "clarity": 0.770},
        "grpo-summarization-length-quality-meteor-bleu":  {"composite": 2.664, "faithfulness": 0.792, "coverage": 0.468, "conciseness": 0.648, "clarity": 0.756},
        "grpo-summarization-length-quality-meteor-rouge": {"composite": 2.769, "faithfulness": 0.832, "coverage": 0.511, "conciseness": 0.659, "clarity": 0.767},
    },
}

REWARD_LABELS = {
    "grpo-summarization-length-only":                 "length penalty only (baseline)",
    "grpo-summarization-length-quality-bleu":         "length + BLEU",
    "grpo-summarization-length-quality-rouge":        "length + ROUGE",
    "grpo-summarization-length-quality-meteor":       "length + METEOR",
    "grpo-summarization-length-quality-bleu-rouge":   "length + BLEU + ROUGE",
    "grpo-summarization-length-quality-meteor-bleu":  "length + METEOR + BLEU",
    "grpo-summarization-length-quality-meteor-rouge": "length + METEOR + ROUGE ⭐ best",
}


def fmt(v):
    return f"{v:.3f}" if v is not None else "—"


def build_readme(repo_id: str, run_name: str, prefix: str, base_model: str, gen_cfg_source: str | None) -> str:
    r = RESULTS.get(prefix, {}).get(run_name, {})
    reward = REWARD_LABELS.get(run_name, run_name)
    gen_cfg_line = f"\n  generation_config_source: {gen_cfg_source}" if gen_cfg_source else ""
    scores_block = (
        f"\n| Faithfulness | Coverage | Conciseness | Clarity | Composite |\n"
        f"|:---:|:---:|:---:|:---:|:---:|\n"
        f"| {fmt(r.get('faithfulness'))} | {fmt(r.get('coverage'))} | {fmt(r.get('conciseness'))} | {fmt(r.get('clarity'))} | {fmt(r.get('composite'))} |\n"
        f"\nEvaluated on 200 examples · judge: `gpt-5-mini-2025-08-07` via DeepEval G-Eval (5 rounds averaged) · "
        f"[full rollouts](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo)\n"
    )

    return f"""\
---
base_model: {base_model}
datasets:
- mlabonne/smoltldr
language:
- en
tags:
- grpo
- summarization
- reinforcement-learning
- mlx
license: apache-2.0
---

# {repo_id.split("/")[-1]}

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) with **GRPO**
on [`mlabonne/smoltldr`](https://huggingface.co/datasets/mlabonne/smoltldr) Reddit summarization,
trained with [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster) on an Apple Silicon Mac cluster.

**Reward:** {reward}

## G-Eval scores (G-Eval, each 0–1; Composite max 4.0)
{scores_block}
## Usage (MLX)

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")
messages = [
    {{"role": "system", "content": "Summarize the following Reddit post in 2-3 sentences."}},
    {{"role": "user",   "content": "<paste Reddit post here>"}},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False))
```

## Training Details

| Setting | Value |
|---|---|
| Base model | [{base_model}](https://huggingface.co/{base_model}) |
| Algorithm | GRPO |
| Dataset | `mlabonne/smoltldr` (train split) |
| Reward | {reward} |
| Hardware | Apple Silicon Mac mini cluster |
| Framework | [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster) (MLX) |
| Weights format | MLX safetensors (bf16) |
"""


api = HfApi()

for model in MODELS:
    ckpt_dir: Path = model["ckpt_dir"]
    prefix: str = model["repo_prefix"]
    base_model: str = model["base_model"]
    gen_cfg_source = model["generation_config_source"]

    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        latest_dir = run_dir / "latest"
        if not latest_dir.exists():
            print(f"  SKIP (no latest/): {run_dir}")
            continue

        repo_id = f"{HF_USER}/{prefix}-{run_dir.name}"
        print(f"\n→ {repo_id}")

        # Create repo
        create_repo(repo_id, repo_type="model", exist_ok=True)

        # Upload weights + config files (flat into repo root)
        api.upload_folder(
            folder_path=str(latest_dir),
            path_in_repo="",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {prefix} {run_dir.name} checkpoint (MLX bf16)",
        )

        # Upload model card
        readme = build_readme(repo_id, run_dir.name, prefix, base_model, gen_cfg_source)
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

        print(f"  ✓ https://huggingface.co/{repo_id}")

print("\nAll done!")
