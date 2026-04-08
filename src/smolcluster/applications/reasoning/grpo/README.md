# GRPO in Smolcluster

This directory contains Smolcluster's MLX-based implementation of Group Relative Policy Optimization (GRPO) for reasoning-style tasks such as GSM8K and summarization.

The implementation is organized around one training process that updates the policy model locally and one or more rollout workers that generate completions from a serving model. The training loop scores groups of completions, converts rewards into within-group advantages, and updates the policy with a clipped objective.

## What This Implementation Does

At a high level, each GRPO step in this repository looks like this:

1. Sample a batch of prompts.
2. Generate multiple rollouts per prompt.
3. Score each rollout with task-specific reward functions.
4. Normalize rewards within each prompt group to produce advantages.
5. Compute per-rollout completion log-probabilities under the current policy.
6. Apply a PPO-style clipped objective, optionally with a KL penalty.
7. Update the policy on GPU via MLX.
8. Periodically save checkpoints and optionally sync rollout workers to the newest policy.

The default GRPO configuration lives in:

- `src/smolcluster/configs/inference/reasoning/grpo/config.yaml`

The main training entry points are:

- `src/smolcluster/applications/reasoning/grpo/train_gsm8k.py`
- `src/smolcluster/applications/reasoning/grpo/train_summarization.py`

Read more about GRPO [here](https://www.alphaxiv.org/abs/2402.03300)
Important implementation detail:

- The model scores only completion tokens, not prompt tokens, when computing rollout log-probabilities.

## Rewarding in the Default GSM8K Setup

The default GSM8K training path combines several simple reward terms.

- `answer_reward`: whether the predicted numeric answer matches the target
- `think_reward`: whether the model used reasoning tags such as `<think>`
- `formatted_reward`: whether the model followed the expected output structure

The default total reward in `train_gsm8k.py` is:

$$
r = r_{\text{answer}} + 0.1 \cdot r_{\text{think}} + 0.1 \cdot r_{\text{format}}
$$

So a fully correct, properly formatted response with reasoning tags gets a maximum default reward of $1.2$.

Summarization uses a separate data loader and reward functions, but the GRPO training pattern is the same.

## Rewarding in the Summarization Setup

The summarization training path (`train_summarization.py`) uses four composable reward signals, all implemented in `rewards/summarization_rewards.py`.

| Reward | Function | Range | What it measures |
|---|---|---|---|
| `rouge_l` | `calculate_rouge_l_reward` | [0, 1] | ROUGE-L F1 vs reference — fluency and phrase ordering via longest-common-subsequence overlap |
| `format` | `calculate_format_reward` | {0, 1} | negative of how far the output deviates from the expected format length |

The default total reward is a weighted sum:

$$
r =  \cdot r_{\text{ROUGE-L}} + \cdot r_{\text{format}}
$$


## Folder Structure

```text
src/smolcluster/applications/reasoning/grpo/
├── README.md                  # This guide
├── train_gsm8k.py             # GRPO training entry point for GSM8K
├── train_summarization.py     # GRPO training entry point for summarization
├── data/
│   ├── gsm8k.py               # GSM8K data loading and prompt prep
│   └── summarization.py       # Summarization data loading and prompt prep
├── rewards/
│   ├── math_rewards.py        # GSM8K-style reward helpers
│   └── summarization_rewards.py
├── evaluation/
│   └── evaluate_gsm8k.py      # pass@k and checkpoint-comparison evaluation
├── scripts/
│   └── launch_grpo_train.sh   # tmux + vLLM launcher with health checks
└── utils/
    ├── amp.py                 # mixed precision helpers
    ├── rollouts.py            # rollout requests to inference workers
    ├── training_utils.py      # tokenization, batching, device helpers
    └── worker_sync.py         # checkpoint sync and worker reload logic
```

Related config files outside this directory:

- `src/smolcluster/configs/inference/reasoning/grpo/config.yaml`
- `src/smolcluster/configs/inference/model_config_inference.yaml`
- `src/smolcluster/configs/inference/cluster_config_inference.yaml`

Related checkpoints:

- `checkpoints/grpo/step_0/`
- `checkpoints/grpo/step_<N>/`
- `checkpoints/grpo/latest/`

`checkpoints/grpo/latest/` is the stable rolling checkpoint that gets overwritten during periodic saves. It is intended to represent the most recently synced policy.

## Important Config Knobs

The most important GRPO settings live in `config.yaml`.

- `device`: MLX device, usually `gpu` on Apple Silicon
- `dtype`: `float32` or `bfloat16`
- `num_epochs`: number of passes over the dataset
- `batch_size`: prompts per optimization batch
- `num_rollouts`: completions generated per worker per prompt
- `max_input_tokens`: maximum prompt-plus-completion token budget used during training tokenization
- `use_kl`: whether to include the reference-model KL term
- `kl_beta`: KL penalty weight
- `clip_ratio`: PPO clipping parameter
- `grad_checkpoint`: enables MLX gradient checkpointing for lower memory usage
- `grad_chunk_size`: prompt chunk size during backward computation
- `rollout_grad_chunk`: rollout chunk size during backward computation
- `force_lora`: force LoRA training even when not strictly required
- `weight_sync.save_every_steps`: how often to refresh `checkpoints/grpo/latest`
- `weight_sync.sync_steps`: how often to push fresh weights to rollout workers
- `vllm`: whether rollout generation is delegated to vLLM workers

## File Map for Further Reading

- `train_gsm8k.py`: reward computation, advantage normalization, GRPO loss, training loop
- `train_summarization.py`: same GRPO loop for summarization
- `utils/rollouts.py`: rollout request fan-out to workers
- `utils/worker_sync.py`: periodic checkpoint save and worker reload
- `evaluation/evaluate_gsm8k.py`: sampled evaluation and checkpoint comparison
