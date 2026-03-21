import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from smolcluster.applications.reasoning.grpo.data.grpo_data import build_train_val_dataloaders
from smolcluster.applications.reasoning.grpo.rewards import (
    calculate_answer_reward,
    calculate_formatted_reward,
)
from smolcluster.applications.reasoning.grpo.rollouts import generate_rollouts_vllm


logger = logging.getLogger(__name__)


with open("configs/inference/reasoning/grpo/config.yaml") as f:
    grpo_config = yaml.safe_load(f)

with open("configs/inference/model_config_inference.yaml") as f:
    model_config = yaml.safe_load(f)


PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the answer. The reasoning process "
    "is enclosed within <think>...</think> and the final answer must be written as "
    "### <answer>. User: {question}. Assistant:"
)




def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    rewards_std = rewards.std()
    return (rewards - rewards.mean()) / rewards_std


def _sequence_logprobs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()

    token_logprobs = torch.log_softmax(shift_logits, dim=-1)
    gathered = token_logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    masked_logprobs = gathered * shift_mask
    token_counts = shift_mask.sum(dim=1).clamp(min=1.0)
    return masked_logprobs.sum(dim=1) / token_counts


def compute_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _sequence_logprobs_from_logits(outputs.logits, input_ids, attention_mask)


def parse_numeric_answer(text: str) -> float:
    matches = re.findall(r"###\s*([-+]?\d*\.?\d+)", text)
    if not matches:
        return float("nan")
    return float(matches[-1])


def compute_grpo_loss(
    ref_logprobs: torch.Tensor,
    curr_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: Dict[str, Any],
) -> torch.Tensor:
    logprobs_ratio = torch.exp(curr_logprobs - ref_logprobs)
    clipped_ratio = torch.clip(
        logprobs_ratio,
        1 - config["clip_ratio"],
        1 + config["clip_ratio"],
    )
    kl = (ref_logprobs - curr_logprobs) - torch.exp(ref_logprobs - curr_logprobs) - 1
    return -torch.mean(
        torch.min(logprobs_ratio * advantages, clipped_ratio * advantages)
        - config["kl_beta"] * kl
    )


def organize_rollouts(rollouts: Dict[int, List[str]]) -> List[str]:
    organized: List[str] = []
    for _worker_rank, generated_texts in sorted(rollouts.items()):
        for generated_text in generated_texts:
            if generated_text is not None:
                organized.append(generated_text)
    return organized


def build_batched_rollout_texts(
    questions: List[str],
    true_answers: List[str],
    config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    rollout_texts: List[str] = []
    rollout_targets: List[str] = []

    for question, true_answer in zip(questions, true_answers):
        prompt = PROMPT.format(question=question)
        worker_rollouts = generate_rollouts_vllm(
            prompt,
            config["num_workers"],
            config["decoding_strategy"],
            config["max_tokens"],
        )
        prompt_rollouts = organize_rollouts(worker_rollouts)
        rollout_texts.extend(prompt_rollouts)
        rollout_targets.extend([true_answer] * len(prompt_rollouts))

    return rollout_texts, rollout_targets


def tokenize_rollouts(
    tokenizer: Any,
    rollout_texts: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenizer(
        rollout_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)


def compute_rewards(
    rollout_texts: List[str],
    rollout_targets: List[str],
    device: torch.device,
) -> torch.Tensor:
    reward_values: List[float] = []

    for generated_text, target_answer in zip(rollout_texts, rollout_targets):
        predicted_numeric_answer = parse_numeric_answer(generated_text)
        target_numeric_answer = parse_numeric_answer(str(target_answer))
        answer_reward = calculate_answer_reward(predicted_numeric_answer, target_numeric_answer)
        format_reward = calculate_formatted_reward(generated_text)
        reward_values.append(float(answer_reward + format_reward))

    return torch.tensor(reward_values, dtype=torch.float32, device=device)


def compute_grad_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def evaluate_batch(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    questions: List[str],
    true_answers: List[str],
) -> Optional[Dict[str, float]]:
    rollout_texts, rollout_targets = build_batched_rollout_texts(questions, true_answers, config)
    if not rollout_texts:
        return None

    device = next(model.parameters()).device
    input_ids, attention_mask = tokenize_rollouts(tokenizer, rollout_texts, device)

    curr_logprobs = compute_logprobs(model, input_ids, attention_mask)
    ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask)
    rewards = compute_rewards(rollout_texts, rollout_targets, device)
    advantages = compute_advantages(rewards)
    loss = compute_grpo_loss(ref_logprobs, curr_logprobs, advantages, config)

    return {
        "loss": float(loss.detach().item()),
        "reward": float(rewards.mean().detach().item()),
        "num_rollouts": float(len(rollout_texts)),
    }


def train_step(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    questions: List[str],
    true_answers: List[str],
) -> Optional[Dict[str, float]]:
    rollout_texts, rollout_targets = build_batched_rollout_texts(questions, true_answers, config)
    if not rollout_texts:
        return None

    device = next(model.parameters()).device
    input_ids, attention_mask = tokenize_rollouts(tokenizer, rollout_texts, device)

    curr_logprobs = compute_logprobs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask)

    rewards = compute_rewards(rollout_texts, rollout_targets, device)
    advantages = compute_advantages(rewards)
    loss = compute_grpo_loss(ref_logprobs, curr_logprobs, advantages, config)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = compute_grad_norm(model)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "reward": float(rewards.mean().detach().item()),
        "grad_norm": float(grad_norm),
        "num_rollouts": float(len(rollout_texts)),
    }


def validate(
    model: Any,
    ref_model: Any,
    config: Dict[str, Any],
    val_dataloader: torch.utils.data.DataLoader,
    tokenizer: Any,
    global_step: int,
) -> Dict[str, float]:

    model.eval()

    loss_values: List[float] = []
    reward_values: List[float] = []
    rollout_counts: List[float] = []

    logger.info("Running evaluation at step %s", global_step)
    with torch.no_grad():
        for questions, true_answers in val_dataloader:
            metrics = evaluate_batch(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                config=config,
                questions=list(questions),
                true_answers=list(true_answers),
            )
            if metrics is None:
                logger.warning("No metrics were returned. Skipping logging of metrics for this batch.")
                continue
            loss_values.append(metrics["loss"])
            reward_values.append(metrics["reward"])
            rollout_counts.append(metrics["num_rollouts"])

    
    model.train()

    return {
        "val/loss": sum(loss_values) / len(loss_values),
        "val/reward": sum(reward_values) / len(reward_values),
        "val/num_rollouts": sum(rollout_counts) / len(rollout_counts),
    }


def train(
    model: Any,
    ref_model: Any,
    config: Dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
) -> None:
    model.train()
    global_step = 0
    total_epochs = config["num_epochs"]
    val_steps = config["val_steps"]

    logger.info("kicking off training for %s epochs", total_epochs)

    for epoch in range(total_epochs):
        epoch_bar = tqdm(
            train_dataloader,
            desc=f"epoch {epoch + 1}/{total_epochs}",
            leave=True,
        )

        for questions, true_answers in epoch_bar:
            global_step += 1
            metrics = train_step(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                config=config,
                questions=list(questions),
                true_answers=list(true_answers),
            )

            if metrics is None:
                logger.info("step %s had no usable rollouts, skipping it", global_step)
                continue

            epoch_bar.set_postfix(
                epoch=epoch + 1,
                step=global_step,
                loss=f"{metrics['loss']:.4f}",
                grad_norm=f"{metrics['grad_norm']:.4f}",
            )
            wandb.log(
                {
                    "train/loss": metrics["loss"],
                    "train/reward": metrics["reward"],
                    "train/grad_norm": metrics["grad_norm"],
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                    "train/num_rollouts": metrics["num_rollouts"],
                },
                step=global_step,
            )

            if global_step % val_steps == 0:
                val_metrics = validate(
                    model=model,
                    ref_model=ref_model,
                    config=config,
                    val_dataloader=val_dataloader,
                    tokenizer=tokenizer,
                    global_step=global_step,
                )
               
                wandb.log(val_metrics, step=global_step)

        epoch_bar.close()


def main() -> None:
    
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_dataloader, val_dataloader = build_train_val_dataloaders(
        grpo_config["batch_size"],
        grpo_config["data"],
    )

    device = torch.device(grpo_config["device"])

    model: Any = AutoModelForCausalLM.from_pretrained(model_config["dp"]["hf_model_name"])
    model = model.to(device)
    ref_model: Any = deepcopy(model)
    ref_model = ref_model.to(device)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=grpo_config["learning_rate"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["dp"]["hf_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wandb.init(
        project=grpo_config.get("wandb_project", "smolcluster-grpo"),
        config=grpo_config,
    )

    
    train(
            model=model,
            ref_model=ref_model,
            config=grpo_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            tokenizer=tokenizer,
        )
    
    wandb.finish()


if __name__ == "__main__":
    main()
