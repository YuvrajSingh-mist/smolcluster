"""GRPO rollout generation — queries vLLM workers to sample candidate responses for each prompt."""
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import requests
import yaml
from smolcluster.utils.logging_utils import emit_smol_event

logger = logging.getLogger(__name__)

lock = threading.Lock()
debug_lock = threading.Lock()

# Resolve config paths relative to project root
_module_dir = Path(__file__).parent
_smolcluster_root = _module_dir.parents[3]  # Navigate from utils -> grpo -> reasoning -> applications -> smolcluster
_config_path = _smolcluster_root / "configs" / "inference" / "reasoning" / "grpo" / "config.yaml"
_cluster_config_path = _smolcluster_root / "configs" / "inference" / "cluster_config_inference.yaml"
_project_root = _smolcluster_root.parent.parent
_debug_dir = _project_root / ".grpo_debug"
_debug_dir.mkdir(parents=True, exist_ok=True)
_debug_log_path = _debug_dir / "vllm_rollouts.jsonl"

with open(_config_path) as f:
    grpo_config = yaml.safe_load(f)

with open(_cluster_config_path) as f:
    cluster_config = yaml.safe_load(f)

API_URL = grpo_config.get("api_url")
NUM_ROLLOUTS = grpo_config["num_rollouts"]


def _get_debug_log_path() -> Path:
    override = os.getenv("SMOLCLUSTER_VLLM_DEBUG_LOG_PATH")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = _project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    _debug_dir.mkdir(parents=True, exist_ok=True)
    return _debug_log_path


def append_vllm_debug_log(record: Dict[str, Any]) -> None:
    debug_log_path = _get_debug_log_path()
    with debug_lock:
        with debug_log_path.open("a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")


def extract_generated_text(result: Dict[str, Any]) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    first_choice = choices[0] or {}
    return first_choice.get("text") or ""

def query(url: str, payload: Dict) -> Dict:
    """Query the inference API."""
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def build_vllm_worker_urls(config: Dict[str, Any]) -> Dict[int, str]:
    """Build OpenAI-compatible vLLM completion URLs for each configured worker rank.
    Workers and host_ip are read from cluster_config_inference.yaml."""
    vllm_cluster = config["vllm_cluster"]
    port = vllm_cluster.get("port", 8000)
    completion_path = vllm_cluster.get("completion_path", "/v1/completions")

    host_ip = cluster_config["host_ip"]
    num_workers = cluster_config["num_workers"]
    workers = cluster_config["workers"]["regular"][:num_workers]

    worker_urls: Dict[int, str] = {}
    for worker in workers:
        hostname = worker["hostname"]
        rank = worker["rank"]
        worker_urls[rank] = f"http://{host_ip[hostname]}:{port}{completion_path}"

    return worker_urls


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
    if API_URL is None:
        raise ValueError("api_url must be set in the GRPO config to use generate_rollouts")

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
    decoding_strategy,
    max_tokens,
) -> Dict[int, list[str]]:
    """
    Generate NUM_ROLLOUTS outputs from each worker for a prompt.
    
    Returns:
        Dict mapping worker_rank -> list of NUM_ROLLOUTS generated texts
    """
    return generate_rollouts_for_prompt(
        prompt, num_workers, decoding_strategy, max_tokens
    )


def _fetch_n_from_worker(
    vllm_url: str,
    worker_rank: int,
    prompt: str,
    n: int,
    rollouts: Dict[int, List[str]],
    lock: threading.Lock,
    max_tokens: int,
    step: Optional[int] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Send ONE request to a vLLM worker asking for n completions.
    vLLM shares the prompt KV cache across all n continuations, which is
    significantly more efficient than n separate single-completion requests.
    """
    api_params = {"prompt": prompt, "max_tokens": max_tokens, "n": n}
    if sampling_params:
        api_params.update(sampling_params)
    logger.info("[vllm worker %d] Requesting n=%d completions from %s", worker_rank, n, vllm_url)
    emit_smol_event("rollout", "out", "grpo")

    try:
        response = requests.post(vllm_url, json=api_params, timeout=300)
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices") or []

        non_empty = [c.get("text", "") for c in choices if c and c.get("text", "").strip()]

        append_vllm_debug_log({
            "step": step,
            "worker_rank": worker_rank,
            "url": vllm_url,
            "n": n,
            "max_tokens": max_tokens,
            "sampling_params": sampling_params or {},
            "prompt_preview": prompt,
            "status_code": response.status_code,
            "num_choices": len(choices),
            "num_non_empty": len(non_empty),
            "texts_preview": {i: t for i, t in enumerate(non_empty)},
        })

        if non_empty:
            emit_smol_event("rollout", "in", "grpo")
            logger.info("[vllm worker %d] Got %d/%d non-empty completion(s)", worker_rank, len(non_empty), n)
            for idx, text in enumerate(non_empty):
                logger.info(
                    "[vllm worker %d | sample %d] %.120s",
                    worker_rank, idx + 1, text.replace("\n", " "),
                )
            with lock:
                rollouts[worker_rank] = non_empty
            return

        emit_smol_event("rollout", "in", "grpo")
        logger.warning("[vllm worker %d] All %d completions empty", worker_rank, n)
    except Exception as e:
        emit_smol_event("rollout", "in", "grpo")
        append_vllm_debug_log(
            {
                "step": step,
                "worker_rank": worker_rank,
                "url": vllm_url,
                "sampling_params": sampling_params or {},
                "error": str(e),
            }
        )
        logger.error("[vllm worker %d] Request failed: %s", worker_rank, e)

    with lock:
        rollouts[worker_rank] = []


def generate_rollouts_vllm(
    prompt: str,
    max_tokens: int,
    num_rollouts: Optional[int] = None,
    step: Optional[int] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[str]]:
    """
    Generate rollouts using worker-node vLLM OpenAI-compatible completion endpoints.

    Sends ONE request per worker with n=num_rollouts, so vLLM shares the prompt
    KV cache across all completions (much more efficient than N separate requests).

    Returns:
        Dict mapping worker rank -> list of generated text strings.
    """
    effective_n: int = int(NUM_ROLLOUTS if num_rollouts is None else num_rollouts)

    _worker_urls = build_vllm_worker_urls(grpo_config)
    logger.info(
        "[generate_rollouts_vllm] Requesting n=%d completions from each of %d worker(s): %s",
        effective_n, len(_worker_urls), list(_worker_urls.keys()),
    )
    logger.info("[generate_rollouts_vllm] Prompt (first 120 chars): %.120s", prompt.replace('\n', ' '))

    rollouts: Dict[int, List[str]] = {}
    vllm_lock = threading.Lock()

    # One thread per worker — each sends a single request with n completions.
    threads = [
        threading.Thread(
            target=_fetch_n_from_worker,
            args=(url, rank, prompt, effective_n, rollouts, vllm_lock, max_tokens, step, sampling_params),
        )
        for rank, url in _worker_urls.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(len(items) for items in rollouts.values())
    logger.info("[generate_rollouts_vllm] All workers done. %d non-empty rollout(s) collected", total)
    return rollouts


# ---------------------------------------------------------------------------
# Rollout orchestration (prompt-level)
# ---------------------------------------------------------------------------

def organize_rollouts(
    rollouts: Dict[int, List[str]],
) -> List[str]:
    """Flatten worker rollouts into an ordered list, dropping empty completions."""
    texts: List[str] = []
    for _rank, items in sorted(rollouts.items()):
        texts.extend(t for t in items if t and t.strip())
    return texts


def _fetch_for_prompt(
    idx: int,
    prompt: str,
    true_answer: str,
    config: Dict[str, Any],
    step: Optional[int] = None,
) -> Tuple[int, List[str], str]:
    """Fetch vLLM rollouts for a single (pre-formatted) prompt. Called concurrently."""
    worker_rollouts = generate_rollouts_vllm(
        prompt,
        max_tokens=config["max_output_tokens"],
        num_rollouts=config["num_rollouts"],
        step=step,
        sampling_params=config.get("vllm_request_overrides"),
    )
    texts = organize_rollouts(worker_rollouts)
    return idx, texts, true_answer


def build_rollouts_per_prompt(
    prompts: List[str],
    true_answers: List[str],
    config: Dict[str, Any],
    step: Optional[int] = None,
) -> List[Tuple[List[str], str]]:
    """Dispatch rollout generation for all prompts concurrently, return per-prompt.

    Returns:
        List of (rollout_texts, true_answer) — one entry per prompt, in input order.
        Prompts that produced zero usable rollouts have an empty rollout_texts list.
    """
    n = len(prompts)
    ordered: List[Optional[Tuple[List[str], str]]] = [None] * n

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(_fetch_for_prompt, idx, prompt, ans, config, step): idx
            for idx, (prompt, ans) in enumerate(zip(prompts, true_answers))
        }
        for future in as_completed(futures):
            idx, prompt_rollouts, true_answer = future.result()
            logger.info(
                "[rollout %d/%d] Received %d usable rollout(s) from workers",
                idx + 1, n, len(prompt_rollouts),
            )
            for r_idx, text in enumerate(prompt_rollouts):
                logger.info(
                    "[rollout %d/%d | sample %d] %.200s",
                    idx + 1, n, r_idx + 1, text.replace("\n", " "),
                )
            ordered[idx] = (prompt_rollouts, true_answer)

    return [
        item if item is not None else ([], true_answers[i])
        for i, item in enumerate(ordered)
    ]


def build_batched_rollout_texts(
    prompts: List[str],
    true_answers: List[str],
    config: Dict[str, Any],
    step: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Dispatch rollout generation for all prompts concurrently, collect in order.

    Returns:
        (rollout_texts, rollout_targets, rollout_questions) — one entry per usable
        rollout across all prompts.
    """
    per_prompt = build_rollouts_per_prompt(prompts, true_answers, config, step=step)
    rollout_texts: List[str] = []
    rollout_targets: List[str] = []
    rollout_questions: List[str] = []
    for (prompt_rollouts, true_answer), prompt in zip(per_prompt, prompts):
        rollout_texts.extend(prompt_rollouts)
        rollout_targets.extend([true_answer] * len(prompt_rollouts))
        rollout_questions.extend([prompt] * len(prompt_rollouts))
    return rollout_texts, rollout_targets, rollout_questions