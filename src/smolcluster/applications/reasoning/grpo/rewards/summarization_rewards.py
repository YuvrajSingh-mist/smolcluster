"""Reward functions for summarization tasks."""

import logging
import threading
from typing import Any, Dict, Optional

import evaluate
from rouge_score import rouge_scorer as _rouge_scorer

logger = logging.getLogger(__name__)

# ROUGE is stateless and thread-safe.
_rouge = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# evaluate/BLEU is thread-safe via thread-local instances.
_local = threading.local()

# NLTK wordnet (used by METEOR) is NOT thread-safe — serialize all meteor calls.
_meteor_lock = threading.Lock()
_meteor_metric = evaluate.load("meteor")


def _get_bleu():
    if not hasattr(_local, "bleu"):
        _local.bleu = evaluate.load("bleu")
    return _local.bleu


def calculate_summary_quality(
    predicted: str,
    reference: str,
    use_rouge: bool = True,
    use_meteor: bool = True,
    use_bleu: bool = True,
) -> Dict[str, float]:
    """Compute individual summary quality scores.

    Returns a dict with keys ``rouge_l``, ``meteor``, ``bleu`` (only for
    enabled metrics). Values are in [0, 1]. Higher is better.
    """
    if not predicted.strip() or not reference.strip():
        scores = {}
        if use_rouge:
            scores["rouge_l"] = 0.0
        if use_meteor:
            scores["meteor"] = 0.0
        if use_bleu:
            scores["bleu"] = 0.0
        return scores

    scores = {}

    if use_rouge:
        scores["rouge_l"] = float(_rouge.score(reference, predicted)["rougeL"].fmeasure)

    if use_meteor:
        with _meteor_lock:
            scores["meteor"] = float(_meteor_metric.compute(predictions=[predicted], references=[reference])["meteor"])

    if use_bleu:
        scores["bleu"] = float(_get_bleu().compute(predictions=[predicted], references=[[reference]])["bleu"])

    return scores


def calculate_length_reward(
    predicted_answer: str,
    max_length: int,
    tokenizer: Optional[Any] = None,
) -> float:
    """Reward based on proximity to a target length.

    Uses token count when a tokenizer is provided, otherwise character count.

    Returns a value in (-1, 0], where 0 means exactly max_length.
    """
    if tokenizer is not None:
        hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
        length = len(hf_tok.encode(predicted_answer, add_special_tokens=False))
        
    # length = len(predicted_answer)
    return -((abs(length - max_length)) / max_length)
