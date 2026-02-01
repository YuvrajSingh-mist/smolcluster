import torch
from transformers import AutoTokenizer


def greedy_decode(activations: torch.Tensor, temperature: float = 1.0) -> str:
    """
    Generate text from the final activations using greedy decoding.
    """
    next_token_logits = activations / temperature
    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    next_token_id = torch.argmax(next_token_probs, dim=-1, keepdim=True)

    return next_token_id


def top_k_sampling(logits: torch.Tensor, top_k: int = 50) -> torch.Tensor:
    """
    Perform top-k sampling on logits.

    Args:
        logits: Token logits of shape (batch_size, vocab_size)
        top_k: Number of top tokens to consider

    Returns:
        Sampled token indices of shape (batch_size, 1)
    """
    # Get top-k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

    # Sample from the top-k distribution
    probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
    next_token_top_k_idx = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    next_token_id = torch.gather(top_k_indices, -1, next_token_top_k_idx)

    return next_token_id


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on logits.

    Args:
        logits: Token logits of shape (batch_size, vocab_size)
        top_p: Cumulative probability threshold for nucleus sampling

    Returns:
        Sampled token indices of shape (batch_size, 1)
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute softmax probabilities
    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Set logits to -inf for tokens to remove
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Sample from the filtered distribution
    probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    next_token_sorted_idx = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    next_token_id = torch.gather(sorted_indices, -1, next_token_sorted_idx)

    return next_token_id


def sample_next_token(
    activations: torch.Tensor,
    tokenized_prompt: torch.Tensor,
    temperature: float,
    tokenizer: AutoTokenizer,
    decoding_strategy: str = "greedy",
    top_p: float = 0.9,
    top_k: int = 50,
) -> tuple[torch.Tensor, bool]:
    """
    Sample next token from final activations and append to prompt.

    Args:
        activations: Model output activations (on device)
        tokenized_prompt: Current tokenized prompt (on CPU)
        temperature: Sampling temperature
        tokenizer: Tokenizer for EOS token check
        decoding_strategy: "greedy", "sampling", "top_p", or "top_k"
        top_p: Nucleus sampling threshold (only used if decoding_strategy="top_p")
        top_k: Number of top tokens (only used if decoding_strategy="top_k")

    Returns:
        (updated_prompt, should_stop)
    """
    next_token_logits = activations[:, -1, :] / temperature

    if decoding_strategy == "greedy":
        next_token_id = greedy_decode(next_token_logits, temperature)

    elif decoding_strategy == "top_p":
        next_token_id = top_p_sampling(next_token_logits, top_p=top_p)

    elif decoding_strategy == "top_k":
        next_token_id = top_k_sampling(next_token_logits, top_k=top_k)

    elif decoding_strategy == "sampling":
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(next_token_probs, num_samples=1)
    else:
        raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

    # Move next_token_id to CPU to concatenate with tokenized_prompt
    tokenized_prompt = torch.cat((tokenized_prompt, next_token_id.cpu()), dim=1)

    # Check if EOS token
    should_stop = next_token_id.item() == tokenizer.eos_token_id

    return tokenized_prompt, should_stop
