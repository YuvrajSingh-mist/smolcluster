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

## GRPO Basics

GRPO is a group-based policy optimization method. Instead of comparing a completion against an external value function, it compares completions generated for the same prompt.

For a prompt $x_t$, the policy produces a group of $C$ completions $y_{t,1}, \dots, y_{t,C}$. Each completion receives a scalar reward $r_{t,c}$. The implementation then standardizes those rewards within the group:

$$
A_{t,c} = \frac{r_{t,c} - \mu_t}{\max(\sigma_t, 10^{-6})}
$$

where:

$$
\mu_t = \frac{1}{C}\sum_{c=1}^{C} r_{t,c}, \qquad
\sigma_t = \sqrt{\frac{1}{C}\sum_{c=1}^{C} (r_{t,c} - \mu_t)^2}
$$

That is exactly what `compute_advantages()` does in `train_gsm8k.py` and `train_summarization.py`.

## Loss Used in This Repo

The core loss is implemented in `compute_grpo_loss()`.

If rollout-time log-probabilities from the behavior policy are available, the code uses a PPO-style clipped importance ratio:

$$
\rho_{t,c} = \exp\left(\log \pi_\theta(y_{t,c} \mid x_t) - \log \pi_{\text{old}}(y_{t,c} \mid x_t)\right)
$$

$$
L_{t,c}^{\text{clip}} = \min\left(\rho_{t,c} A_{t,c}, \operatorname{clip}(\rho_{t,c}, 1 - \epsilon, 1 + \epsilon) A_{t,c}\right)
$$

If `old_logprobs` are not present, the code falls back to a simpler policy-gradient term:

$$
L_{t,c}^{\text{pg}} = A_{t,c} \log \pi_\theta(y_{t,c} \mid x_t)
$$

If a frozen reference model is present and `use_kl: true`, the code subtracts a KL-style penalty:

$$
\operatorname{KL}_{t,c} = \exp(\log \pi_{\text{ref}} - \log \pi_\theta) - (\log \pi_{\text{ref}} - \log \pi_\theta) - 1
$$

and the per-rollout objective becomes:

$$
L_{t,c} = L_{t,c}^{\text{clip or pg}} - \beta \operatorname{KL}_{t,c}
$$

Finally, the code averages first within each prompt group and then across prompts, and negates the result for minimization:

$$
\mathcal{L}_{\text{GRPO}} = - \frac{1}{T} \sum_{t=1}^{T} \left(\frac{1}{C} \sum_{c=1}^{C} L_{t,c}\right)
$$

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

## How to Train

### Option 1: Use the Launcher

The launcher is the most complete way to run GRPO in this repo because it performs preflight checks, manages tmux sessions, and brings up rollout workers.

Run GSM8K training:

```bash
bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_train.sh gsm8k
```

Run summarization training:

```bash
bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_train.sh summarization
```

Preview commands without executing them:

```bash
bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_train.sh --dry-run gsm8k
```

Clean up the tmux session and worker processes:

```bash
bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_train.sh --cleanup
```

The launcher expects:

- `tmux`
- `ssh`
- `curl`
- `yq`
- valid cluster and model config files
- `vllm: true` in the GRPO config when using the launcher

### Option 2: Run the Training Script Directly

If you want to run the MLX trainer directly without the launcher:

```bash
python src/smolcluster/applications/reasoning/grpo/train_gsm8k.py
```

or:

```bash
python src/smolcluster/applications/reasoning/grpo/train_summarization.py
```

The training scripts read configuration directly from:

```text
src/smolcluster/configs/inference/reasoning/grpo/config.yaml
```

So the normal workflow is:

1. Edit `config.yaml`.
2. Make sure the base model and cluster settings are correct.
3. Start training.

### Device Behavior

Training explicitly resolves the MLX device from the GRPO config and calls `mx.set_default_device(...)` before loading the model.

If you keep:

```yaml
device: "gpu"
```

then GRPO training runs on the MLX GPU path.

## How Rollouts Work

When `vllm: true`, the trainer does not sample rollouts from the local training process. Instead, it calls worker inference endpoints and asks each worker to generate multiple completions.

The rollout utilities:

- read worker topology from `cluster_config_inference.yaml`
- build per-worker completion URLs
- send requests concurrently
- collect generated texts back into prompt groups

This decouples policy optimization from high-throughput text generation.

## Checkpoints

Training can write two kinds of artifacts depending on the model setup.

- Full weights: `model.safetensors`
- LoRA adapters: `adapters/adapters.safetensors` plus `adapter_config.json`

Checkpoint directories follow this pattern:

```text
checkpoints/grpo/
├── latest/
├── step_0/
├── step_100/
└── step_200/
```

In this repo, `latest/` is the rolling checkpoint used for worker syncing and the most recent evaluation target.

## How to Evaluate

The GSM8K evaluator supports two modes.

### 1. pass@k-style sampling evaluation

Evaluate one model with multiple sampled rollouts:

```bash
python src/smolcluster/applications/reasoning/grpo/evaluation/evaluate_gsm8k.py \
  --model-path Qwen/Qwen3-0.6B \
  --num-rollouts 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --device gpu
```

Evaluate the latest saved GRPO checkpoint:

```bash
python src/smolcluster/applications/reasoning/grpo/evaluation/evaluate_gsm8k.py \
  --model-path checkpoints/grpo/latest \
  --num-rollouts 8 \
  --device gpu
```

Evaluate through the same multi-worker vLLM rollout path used during training:

```bash
python src/smolcluster/applications/reasoning/grpo/evaluation/evaluate_gsm8k.py \
  --model-path checkpoints/grpo/latest \
  --use-vllm \
  --num-rollouts 4
```

In `--use-vllm` mode, `num_rollouts` is interpreted per worker. If you have 2 workers and set `--num-rollouts 4`, evaluation requests 8 total completions per prompt.

To start the workers with the exact same setup used by training and then run evaluation in one command, use:

```bash
bash src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_eval.sh passk
```

That launcher:

- reads the same GRPO, cluster, and model configs as training
- starts worker-side vLLM with the same `weight_sync.vllm_start_cmd`
- uses the same health checks as training
- evaluates `checkpoints/grpo/latest` by default through the worker rollout path
- writes worker rollout transport logs to `evaluation/eval-rollouts/vllm_rollouts.jsonl`
- writes rollout answer logs (same schema as `.grpo_debug/rollout_answers.jsonl`) to `evaluation/eval-rollouts/rollout_answers.jsonl`

### 2. before/after checkpoint comparison

Compare `step_0` against the most recent checkpoint discovered under `checkpoints/grpo`:

```bash
python src/smolcluster/applications/reasoning/grpo/evaluation/evaluate_gsm8k.py \
  --checkpoint-dir checkpoints/grpo \
  --device gpu
```

Compare explicit checkpoints:

```bash
python src/smolcluster/applications/reasoning/grpo/evaluation/evaluate_gsm8k.py \
  --step0 checkpoints/grpo/step_0 \
  --final checkpoints/grpo/latest \
  --device gpu
```

The evaluator now sets the MLX default device explicitly, and its default is `--device gpu`.

## Typical Workflow

For GSM8K, a practical loop is:

1. Set the base model and GRPO hyperparameters in `config.yaml`.
2. Launch rollout workers and training with `launch_grpo_train.sh`.
3. Watch `checkpoints/grpo/latest/` for the newest synced policy.
4. Run checkpoint comparison to measure accuracy gains over `step_0`.
5. Tune rewards, `clip_ratio`, `num_rollouts`, or LoRA settings as needed.

## Notes and Caveats

- This implementation uses group-normalized rewards, so extremely low reward variance inside a group will shrink the effective learning signal.
- If `vllm: true`, rollout quality and latency depend on worker health and cluster configuration.
- LoRA is especially useful when training quantized models because it keeps the trainable parameter count small.
- `checkpoints/grpo/latest/` may contain either full weights or adapter-only weights, depending on the active training setup.

## File Map for Further Reading

- `train_gsm8k.py`: reward computation, advantage normalization, GRPO loss, training loop
- `train_summarization.py`: same GRPO loop for summarization
- `utils/rollouts.py`: rollout request fan-out to workers
- `utils/worker_sync.py`: periodic checkpoint save and worker reload
- `evaluation/evaluate_gsm8k.py`: sampled evaluation and checkpoint comparison
