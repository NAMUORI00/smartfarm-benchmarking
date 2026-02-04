"""Retrieval evaluation metrics for RAG systems.

Standard IR metrics:
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that are retrieved  
- MRR: Mean Reciprocal Rank - average of reciprocal ranks of first relevant doc
- NDCG@K: Normalized Discounted Cumulative Gain - ranking quality with graded relevance
- Hit Rate: Whether at least one relevant doc is retrieved
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Precision@K: fraction of top-k retrieved docs that are relevant.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Precision score [0, 1]
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Recall@K: fraction of relevant docs that appear in top-k.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Recall score [0, 1]
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_retrieved = len(top_k & relevant)
    return relevant_retrieved / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant document.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: Set of relevant document IDs
    
    Returns:
        MRR score [0, 1]
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K.
    
    Args:
        relevances: List of relevance scores for retrieved docs
        k: Number of top results to consider
    
    Returns:
        DCG score
    """
    relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevance_scores: Dict mapping doc_id -> relevance score (0-3 typically)
        k: Number of top results to consider
    
    Returns:
        NDCG score [0, 1]
    """
    # Get relevance scores for retrieved docs
    retrieved_relevances = [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved[:k]]
    
    # Ideal ranking (sorted by relevance)
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    
    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved: List[str], relevant: Set[str], k: int = None) -> float:
    """Hit Rate: whether at least one relevant doc is in top-k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Optional limit on top results (None = all)
    
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if k is not None:
        retrieved = retrieved[:k]
    return 1.0 if any(doc_id in relevant for doc_id in retrieved) else 0.0


@dataclass
class RetrievalMetrics:
    """Container for computing multiple retrieval metrics."""
    
    # Paper primary K=4 + standard K=10
    k_values: List[int] = field(default_factory=lambda: [4, 10])
    
    def compute_all(
        self,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute all retrieval metrics.
        
        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            relevance_scores: Optional graded relevance scores for NDCG
        
        Returns:
            Dict of metric_name -> value
        """
        results = {}
        
        # MRR (doesn't depend on k)
        results["mrr"] = mrr(retrieved, relevant)
        
        # K-dependent metrics
        for k in self.k_values:
            results[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            results[f"hit_rate@{k}"] = hit_rate(retrieved, relevant, k)
            
            if relevance_scores:
                results[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevance_scores, k)
        
        return results
    
    def aggregate(self, all_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple queries (mean).
        
        Args:
            all_results: List of per-query metric dicts
        
        Returns:
            Dict of aggregated metrics
        """
        if not all_results:
            return {}
        
        # Get all metric names from first result
        metric_names = list(all_results[0].keys())
        
        aggregated = {}
        for name in metric_names:
            values = [r.get(name, 0.0) for r in all_results]
            aggregated[f"mean_{name}"] = sum(values) / len(values)
        
        return aggregated
