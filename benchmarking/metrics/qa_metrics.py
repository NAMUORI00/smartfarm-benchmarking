"""QA evaluation metrics for RAG systems.

Answer quality metrics:
- Exact Match: Whether answer exactly matches reference
- F1 Score: Token-level F1 between answer and reference
- BLEU: N-gram overlap score
- ROUGE-L: Longest common subsequence based score
- Semantic Similarity: Embedding-based similarity
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional imports for advanced metrics
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return _normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """Exact match score (case-insensitive, normalized).
    
    Args:
        prediction: Model's answer
        reference: Ground truth answer
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score.
    
    Args:
        prediction: Model's answer
        reference: Ground truth answer
    
    Returns:
        F1 score [0, 1]
    """
    pred_tokens = Counter(_tokenize(prediction))
    ref_tokens = Counter(_tokenize(reference))
    
    # Common tokens
    common = pred_tokens & ref_tokens
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / sum(pred_tokens.values()) if pred_tokens else 0.0
    recall = num_common / sum(ref_tokens.values()) if ref_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-gram counts from token list."""
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
    weights: Optional[List[float]] = None,
) -> float:
    """BLEU score (simplified, single reference).
    
    Args:
        prediction: Model's answer
        reference: Ground truth answer
        max_n: Maximum n-gram order
        weights: Weights for each n-gram (default: uniform)
    
    Returns:
        BLEU score [0, 1]
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        
        # Clipped counts
        clipped = sum(min(count, ref_ngrams.get(ngram, 0)) 
                      for ngram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        
        precisions.append(clipped / total if total > 0 else 0.0)
    
    # Geometric mean of precisions (with smoothing)
    import math
    log_sum = 0.0
    for w, p in zip(weights, precisions):
        if p > 0:
            log_sum += w * math.log(p)
        else:
            return 0.0  # Zero precision means zero BLEU
    
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0
    
    return bp * math.exp(log_sum)


def _lcs_length(s1: List[str], s2: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_score(prediction: str, reference: str, rouge_type: str = "L") -> Dict[str, float]:
    """ROUGE score (ROUGE-L by default).
    
    Args:
        prediction: Model's answer
        reference: Ground truth answer
        rouge_type: "L" for ROUGE-L (LCS-based), "1" for unigram, "2" for bigram
    
    Returns:
        Dict with precision, recall, f1
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if rouge_type == "L":
        lcs_len = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
    else:
        n = int(rouge_type)
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        
        overlap = sum(min(pred_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) 
                      for ng in set(pred_ngrams) | set(ref_ngrams))
        
        precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


# Global embedding model (lazy loaded)
_embedding_model = None


def _get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None and HAS_SENTENCE_TRANSFORMERS:
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedding_model


def semantic_similarity(prediction: str, reference: str) -> float:
    """Semantic similarity using sentence embeddings.
    
    Args:
        prediction: Model's answer
        reference: Ground truth answer
    
    Returns:
        Cosine similarity [0, 1] (clamped)
    """
    model = _get_embedding_model()
    if model is None:
        # Fallback to F1 if no embedding model
        return f1_score(prediction, reference)
    
    embeddings = model.encode([prediction, reference])
    
    # Cosine similarity
    import numpy as np
    pred_emb, ref_emb = embeddings[0], embeddings[1]
    similarity = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, float(similarity)))


@dataclass
class QAMetrics:
    """Container for computing multiple QA metrics."""
    
    use_semantic: bool = True
    
    def compute_all(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute all QA metrics.
        
        Args:
            prediction: Model's answer
            reference: Ground truth answer
        
        Returns:
            Dict of metric_name -> value
        """
        results = {
            "exact_match": exact_match(prediction, reference),
            "f1": f1_score(prediction, reference),
            "bleu": bleu_score(prediction, reference),
        }
        
        rouge = rouge_score(prediction, reference, "L")
        results["rouge_l_precision"] = rouge["precision"]
        results["rouge_l_recall"] = rouge["recall"]
        results["rouge_l_f1"] = rouge["f1"]
        
        if self.use_semantic and HAS_SENTENCE_TRANSFORMERS:
            results["semantic_similarity"] = semantic_similarity(prediction, reference)
        
        return results
    
    def aggregate(self, all_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple examples (mean)."""
        if not all_results:
            return {}
        
        metric_names = list(all_results[0].keys())
        aggregated = {}
        for name in metric_names:
            values = [r.get(name, 0.0) for r in all_results]
            aggregated[f"mean_{name}"] = sum(values) / len(values)
        
        return aggregated
