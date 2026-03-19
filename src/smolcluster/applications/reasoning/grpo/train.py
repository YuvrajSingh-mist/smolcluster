from datasets import load_dataset
import torch
import re

from smolcluster.applications.reasoning.grpo.rewards import calculate_reward, calculate_formatted_reward
from smolcluster.applications.reasoning.grpo.rollouts import generate_rollouts


from transformers import AutoTokenizer

with open("configs/inference/reasoning/grpo/config.yaml") as f:
    config = yaml.safe_load(f)


PROMPT = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}. Assistant:"""   


def extract_answer_from_gsm8k(gsm8k_answer: str) -> float:
    """
    Extract the final answer from a GSM8K answer string.

    Args:
        gsm8k_answer: The answer string from GSM8K, which may contain reasoning steps.
    Returns:
        The extracted final answer as a float.
    """
    
    # Use regex to find the final answer in the GSM8K answer string
    match = re.search(r"### (.*)", gsm8k_answer)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Could not extract final answer from GSM8K answer: {gsm8k_answer}")


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute the advantages for a batch of predicted and true answers.

    Args:
        rewards: A tensor of shape (batch_size,) containing the rewards.
    
    Returns:
        A tensor of shape (batch_size,) containing the computed advantages.
    """
    
    advantages = (rewards - rewards.mean()) / rewards.std()
    
    return advantages

def compute_logprobs(tokenizer: AutoTokenizer, predicted_answers: torch.Tensor, prompt: str, question: str) -> torch.Tensor:
    """
    Compute the log probabilities of the predicted answers given the true answers.

    Args:
        predicted_answers: A tensor of shape (batch_size,) containing the predicted answers.
    
    Returns:
        A tensor of shape (batch_size,) containing the log probabilities.
    """
    prompt_idx = torch.tensor([tokenizer.encode(prompt)])

    full_prompt = prompt.format(question=question)
    full_prompt_idx = torch.tensor([tokenizer.encode(full_prompt)])
    
    start_idx = prompt_idx.shape[1]
    end_idx = full_prompt_idx.shape[1]
    
    actual_predicted_answers = predicted_answers[:, start_idx:end_idx]
    
    log_probs = torch.nn.functional.log_softmax(actual_predicted_answers, dim=-1)

    sample_log_probs = torch.mean(log_probs, dim=-1)

    return sample_log_probs


def compute_grpo_loss(ref_logprobs: torch.Tensor, curr_logprobs: torch.Tensor, advantages: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute the GRPO loss for a batch of predicted and true answers.

    Args:
        logprobs: A tensor of shape (batch_size,) containing the log probabilities of the predicted answers.
        advantages: A tensor of shape (batch_size,) containing the computed advantages.
    Returns:
        A tensor containing the GRPO loss.
    """
    logprobs_ratio = torch.exp(curr_logprobs - ref_logprobs)
    kl = (ref_logprobs - curr_logprobs) - torch.exp(ref_logprobs - curr_logprobs) - 1
    loss = -torch.mean(torch.min(logprobs_ratio * advantages, torch.clip(logprobs_ratio, 1 - config["clip_ratio"], 1 + config["clip_ratio"]) * advantages) - config["kl_beta"] * kl)
    
    return loss


train_ds = load_dataset("gsm8k", split="train")
val_ds = load_dataset("gsm8k", split="validation")

questions = train_ds["train"]["question"]
answers = train_ds["train"].map(lambda x: {"answer": extract_answer_from_gsm8k(x["answer"])})["answer"]


def train():
    
    for epoch in range(config["num_epochs"]):
        
        for question, true_answer in zip(questions, answers):
            
            prompt = PROMPT.format(question=question)
            rollouts = generate_rollouts(prompt)
            
            
            reward = calculate_formatted_reward(predicted_answer)
            logprobs = compute_logprobs(tokenizer, predicted_answer, prompt, question)
            advantages = compute_advantages(reward)
            loss = compute_grpo_loss(ref_logprobs, logprobs, advantages, config["kl_beta"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        


