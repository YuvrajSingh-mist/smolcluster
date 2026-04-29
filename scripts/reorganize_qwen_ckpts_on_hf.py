"""Reorganize HF model repo: move root-level Qwen checkpoints under Qwen2.5-0.5B-Instruct-bf16/."""

from pathlib import Path
from huggingface_hub import HfApi, CommitOperationDelete

REPO_ID = "YuvrajSingh9886/reddit-posts-summarization-grpo"
QWEN_CKPT = Path(__file__).parent.parent / "checkpoints" / "Qwen2.5-0.5B-instruct-bf16"

api = HfApi()

# ---------------------------------------------------------------------------
# 1. Upload Qwen checkpoints under Qwen2.5-0.5B-Instruct-bf16/
# ---------------------------------------------------------------------------
print("Uploading Qwen checkpoints to Qwen2.5-0.5B-Instruct-bf16/ ...")
api.upload_folder(
    folder_path=str(QWEN_CKPT),
    path_in_repo="Qwen2.5-0.5B-Instruct-bf16",
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=["**/README.md"],   # skip per-checkpoint READMEs from old upload
    commit_message="Move Qwen2.5 checkpoints under Qwen2.5-0.5B-Instruct-bf16/ subfolder",
)
print("Upload done.")

# ---------------------------------------------------------------------------
# 2. Delete old root-level Qwen folders
# ---------------------------------------------------------------------------
ROOT_QWEN_FOLDERS = [
    "grpo-summarization-length-quality-bleu",
    "grpo-summarization-length-quality-bleu-rouge",
    "grpo-summarization-length-quality-meteor",
    "grpo-summarization-length-quality-meteor-bleu",
    "grpo-summarization-length-quality-meteor-rouge",
    "grpo-summarization-length-quality-rouge",
]

# Collect all files in those folders currently on the repo
print("\nCollecting root-level Qwen files to delete...")
all_files = list(api.list_repo_files(REPO_ID, repo_type="model"))
to_delete = [
    CommitOperationDelete(path_in_repo=f)
    for f in all_files
    if any(f.startswith(folder + "/") for folder in ROOT_QWEN_FOLDERS)
]
print(f"  Deleting {len(to_delete)} files...")

api.create_commit(
    repo_id=REPO_ID,
    repo_type="model",
    operations=to_delete,
    commit_message="Remove old root-level Qwen checkpoint folders",
)
print("Old folders deleted.")

# ---------------------------------------------------------------------------
# 3. Update README to reflect new paths
# ---------------------------------------------------------------------------
print("\nUpdating README...")

new_readme = """\
---
base_model:
  - Qwen/Qwen2.5-0.5B-Instruct
  - liquid-ai/LFM-2.5-350M
datasets:
- mlabonne/smoltldr
language:
- en
tags:
- grpo
- summarization
- qwen2.5
- lfm2
- reinforcement-learning
- mlx
license: apache-2.0
---

# GRPO Summarization — Qwen2.5-0.5B-Instruct & LFM2.5-350M

Checkpoints for two models fine-tuned with **Group Relative Policy Optimization (GRPO)**
on the [`mlabonne/smoltldr`](https://huggingface.co/datasets/mlabonne/smoltldr) Reddit
summarization dataset, trained with [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster)
on an Apple Silicon Mac cluster.

Each checkpoint sub-folder differs only in which automatic quality metric was added
to the length reward during GRPO training.

Full rollouts, per-example scores, and paired significance tests:
[reddit-posts-summarization-grpo (dataset)](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo)

---

## Checkpoints

### Qwen2.5-0.5B-Instruct-bf16

Fine-tuned from [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
Checkpoints live under `Qwen2.5-0.5B-Instruct-bf16/grpo-summarization-*/`.

| Subfolder | Reward | Faithfulness | Coverage | Conciseness | Clarity | Composite |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `grpo-summarization-length-quality-bleu` | length + BLEU | 0.680 | 0.399 | 0.577 | 0.744 | 2.400 |
| `grpo-summarization-length-quality-rouge` | length + ROUGE | — | — | — | — | — |
| `grpo-summarization-length-quality-meteor` | length + METEOR | — | — | — | — | — |
| `grpo-summarization-length-quality-bleu-rouge` | length + BLEU + ROUGE | 0.810 | 0.502 | 0.650 | 0.770 | 2.732 |
| `grpo-summarization-length-quality-meteor-bleu` | length + METEOR + BLEU | 0.792 | 0.468 | 0.648 | 0.756 | 2.664 |
| **`grpo-summarization-length-quality-meteor-rouge`** | **length + METEOR + ROUGE** | **0.832** | **0.511** | **0.659** | **0.767** | **2.769** |

Baseline (length-only, composite 2.416): not included as a checkpoint.

### LFM2.5-350M-bf16

Fine-tuned from [liquid-ai/LFM-2.5-350M](https://huggingface.co/liquid-ai/LFM-2.5-350M).
Checkpoints live under `LFM-2.5-350M-bf16/grpo-summarization-*/`.

| Subfolder | Reward | Faithfulness | Coverage | Conciseness | Clarity | Composite |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `grpo-summarization-length-only` | length only (baseline) | 0.627 | 0.378 | 0.554 | 0.674 | 2.233 |
| `grpo-summarization-length-quality-bleu` | length + BLEU | 0.620 | 0.401 | 0.556 | 0.665 | 2.243 |
| `grpo-summarization-length-quality-rouge` | length + ROUGE | 0.642 | 0.414 | 0.575 | 0.646 | 2.278 |
| `grpo-summarization-length-quality-meteor` | length + METEOR | 0.689 | 0.433 | 0.595 | 0.641 | 2.358 |
| `grpo-summarization-length-quality-bleu-rouge` | length + BLEU + ROUGE | 0.696 | 0.443 | 0.606 | 0.643 | 2.387 |
| `grpo-summarization-length-quality-meteor-bleu` | length + METEOR + BLEU | 0.696 | 0.451 | 0.595 | 0.634 | 2.377 |
| **`grpo-summarization-length-quality-meteor-rouge`** | **length + METEOR + ROUGE** | **0.834** | **0.493** | **0.685** | **0.690** | **2.701** |

Composite = sum of four G-Eval metrics (max 4.0). Evaluated on 200 examples with
`gpt-5-mini-2025-08-07` as the LLM judge (5 rounds averaged).

---

## Usage (MLX)

```bash
git clone https://huggingface.co/YuvrajSingh9886/reddit-posts-summarization-grpo
cd reddit-posts-summarization-grpo
```

### Qwen2.5-0.5B — best run

```python
from mlx_lm import load, generate

model, tokenizer = load("Qwen2.5-0.5B-Instruct-bf16/grpo-summarization-length-quality-meteor-rouge/latest")
messages = [
    {"role": "system", "content": "Summarize the following Reddit post in 2-3 sentences."},
    {"role": "user",   "content": "<paste your Reddit post here>"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
print(output)
```

### LFM2.5-350M — best run

```python
from mlx_lm import load, generate

model, tokenizer = load("LFM-2.5-350M-bf16/grpo-summarization-length-quality-meteor-rouge/latest")
messages = [
    {"role": "system", "content": "Summarize the following Reddit post in 2-3 sentences."},
    {"role": "user",   "content": "<paste your Reddit post here>"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
print(output)
```

---

## Repository Structure

```
reddit-posts-summarization-grpo/
├── README.md
│
├── Qwen2.5-0.5B-Instruct-bf16/
│   ├── grpo-summarization-length-quality-bleu/latest/
│   │   ├── model.safetensors        # MLX bf16 weights (~940 MB)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── chat_template.jinja
│   ├── grpo-summarization-length-quality-bleu-rouge/latest/   └── ...
│   ├── grpo-summarization-length-quality-meteor/latest/       └── ...
│   ├── grpo-summarization-length-quality-meteor-bleu/latest/  └── ...
│   ├── grpo-summarization-length-quality-meteor-rouge/latest/ └── ...  ← best (2.769)
│   └── grpo-summarization-length-quality-rouge/latest/        └── ...
│
└── LFM-2.5-350M-bf16/
    ├── grpo-summarization-length-only/latest/
    │   ├── model.safetensors    # MLX bf16 weights (~709 MB)
    │   ├── config.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── chat_template.jinja
    ├── grpo-summarization-length-quality-bleu/latest/        └── ...
    ├── grpo-summarization-length-quality-rouge/latest/       └── ...
    ├── grpo-summarization-length-quality-meteor/latest/      └── ...
    ├── grpo-summarization-length-quality-bleu-rouge/latest/  └── ...
    ├── grpo-summarization-length-quality-meteor-bleu/latest/ └── ...
    └── grpo-summarization-length-quality-meteor-rouge/latest/ └── ...  ← best (2.701)
```

---

## Training Details

| Setting | Qwen2.5-0.5B | LFM2.5-350M |
|---|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct | liquid-ai/LFM-2.5-350M |
| Algorithm | GRPO | GRPO |
| Dataset | `mlabonne/smoltldr` (train split) | `mlabonne/smoltldr` (train split) |
| Shared reward | Length penalty | Length penalty |
| Variable reward | BLEU / ROUGE / METEOR and combinations | BLEU / ROUGE / METEOR and combinations |
| Hardware | Apple Silicon Mac mini cluster | Apple Silicon Mac mini cluster |
| Framework | [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster) (MLX) | [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster) (MLX) |
| Weights format | MLX safetensors (bf16) | MLX safetensors (bf16) |
| Eval examples | 200 (validation split) | 200 (validation split) |
| Judge | `gpt-5-mini-2025-08-07` via DeepEval GEval | `gpt-5-mini-2025-08-07` via DeepEval GEval |
"""

api.upload_file(
    path_or_fileobj=new_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Update README: Qwen checkpoints now under Qwen2.5-0.5B-Instruct-bf16/",
)
print("README updated.")
print(f"\nDone! https://huggingface.co/{REPO_ID}")
