import requests
from typing import Dict
import threading
import yaml

lock = threading.Lock()

with open("configs/inference/reasoning/grpo/config.yaml") as f:
    grpo_config = yaml.safe_load(f)

API_URL = grpo_config["api_url"]
NUM_ROLLOUTS = grpo_config["num_rollouts"]

def query(url: str, payload: Dict) -> Dict:
    """Query the inference API."""
    response = requests.post(url, json=payload)
    return response.json()


def handle_worker_rollout(
    url: str,
    payload: Dict,
    worker_rank: int,
    rollout_idx: int,
    rollouts: Dict,
    lock: threading.Lock,
) -> None:
    """Generate a single rollout for a worker and store result."""
    response = query(url, payload)
    
    with lock:
        if worker_rank not in rollouts:
            rollouts[worker_rank] = [None] * NUM_ROLLOUTS
        rollouts[worker_rank][rollout_idx] = response["generated_text"]


def generate_rollouts_for_prompt(
    prompt: str,
    num_workers: int,
    decoding_strategy: str,
    max_tokens: int,
) -> Dict[int, list[str]]:
    """Generate NUM_ROLLOUTS outputs from each worker for a single prompt."""
    threads = []
    rollouts = {}
    
    for worker_rank in range(num_workers):
        for rollout_idx in range(NUM_ROLLOUTS):
            
            payload = {
                "text": prompt,
                "worker_rank": worker_rank,
                "max_tokens": max_tokens,
                "decoding_strategy": decoding_strategy,
            }
            
            thread = threading.Thread(
                target=handle_worker_rollout,
                args=(API_URL, payload, worker_rank, rollout_idx, rollouts, lock),
            )
            thread.start()
            threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return rollouts


def generate_rollouts(
    prompt: str,
    num_workers: int,
    decoding_strategy: str = "top_p",
    max_tokens: int = 256,
) -> Dict[int, list[str]]:
    """
    Generate NUM_ROLLOUTS outputs from each worker for a prompt.
    
    Returns:
        Dict mapping worker_rank -> list of NUM_ROLLOUTS generated texts
    """
    return generate_rollouts_for_prompt(
        prompt, num_workers, decoding_strategy, max_tokens
    )


def handle_vllm_rollout(
    vllm_url: str,
    prompt: str,
    rollout_idx: int,
    rollouts: Dict,
    lock: threading.Lock,
    decoding_strategy: str,
    max_tokens: int,
) -> None:
    """Generate a single rollout from vLLM and store result."""
    try:
        # Build OpenAI-compatible request
        api_params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        
        if decoding_strategy == "top_p":
            api_params["top_p"] = 0.9
            api_params["temperature"] = 0.7
        elif decoding_strategy == "temperature":
            api_params["temperature"] = 0.8
            api_params["top_p"] = 1.0
        else:
            api_params["temperature"] = 0.7
        
        response = requests.post(vllm_url, json=api_params, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        generated_text = result.get("choices", [{}])[0].get("text", "")
        
        with lock:
            if "vllm" not in rollouts:
                rollouts["vllm"] = {}
            rollouts["vllm"][rollout_idx] = generated_text
    except Exception as e:
        print(f"Error generating rollout {rollout_idx} from vLLM: {e}")
        with lock:
            if "vllm" not in rollouts:
                rollouts["vllm"] = {}
            rollouts["vllm"][rollout_idx] = ""


def generate_rollouts_vllm(
    prompt: str,
    vllm_completion_url: str,
    decoding_strategy: str = "top_p",
    max_tokens: int = 256,
    num_rollouts: int = None,
) -> Dict[int, list[str]]:
    """
    Generate rollouts using vLLM's OpenAI-compatible API.
    
    Args:
        prompt: The prompt to complete
        vllm_completion_url: OpenAI-compatible completion endpoint (e.g., http://localhost:8000/v1/completions)
        decoding_strategy: "top_p", "temperature", or "greedy"
        max_tokens: Max tokens to generate
        num_rollouts: Number of rollouts to generate (uses grpo_config value if None)
    
    Returns:
        Dict mapping "vllm" -> list of generated texts
    """
    if num_rollouts is None:
        num_rollouts = NUM_ROLLOUTS
    
    threads = []
    rollouts = {}
    vllm_lock = threading.Lock()
    
    for rollout_idx in range(num_rollouts):
        thread = threading.Thread(
            target=handle_vllm_rollout,
            args=(
                vllm_completion_url,
                prompt,
                rollout_idx,
                rollouts,
                vllm_lock,
                decoding_strategy,
                max_tokens,
            ),
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return rollouts