"""QA evaluation metrics (paper-minimal).

We keep only well-known, standard metrics aligned with the paper intro goals:
  - Exact Match (EM)
  - Token-level F1
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return _normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """Exact Match score (normalized, case-insensitive)."""
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    pred_tokens = Counter(_tokenize(prediction))
    ref_tokens = Counter(_tokenize(reference))
    common = pred_tokens & ref_tokens
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / sum(pred_tokens.values()) if pred_tokens else 0.0
    recall = num_common / sum(ref_tokens.values()) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class QAMetrics:
    """Container for computing QA metrics used in the paper."""

    def compute_all(self, prediction: str, reference: str) -> Dict[str, float]:
        return {
            "exact_match": exact_match(prediction, reference),
            "f1": f1_score(prediction, reference),
        }

