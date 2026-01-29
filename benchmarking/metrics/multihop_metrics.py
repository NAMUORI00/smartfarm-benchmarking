"""Multi-hop reasoning evaluation metrics for knowledge graph-based QA.

Evaluates multi-hop reasoning paths through knowledge graphs:
- Hop-level accuracy: Whether each individual hop is correct
- Path exact match: Whether the entire reasoning path matches ground truth
- Supporting facts evaluation: Precision, recall, F1 for intermediate evidence
- Answer quality: Exact match and F1 for final answers

These metrics measure the quality of multi-step reasoning chains where the
model must traverse multiple nodes and edges in a knowledge graph to reach
a final answer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class HopResult:
    """Represents a single hop in a multi-hop reasoning path.

    Attributes:
        hop_index: 0-based index of this hop in the path
        source_node: ID of the source node
        edge_type: Type/relation of the edge traversed
        target_node: ID of the target node reached
        confidence: Confidence score for this hop [0, 1]
    """
    hop_index: int
    source_node: str
    edge_type: str
    target_node: str
    confidence: float = 1.0


@dataclass
class MultihopPath:
    """Represents a complete multi-hop reasoning path.

    Attributes:
        question: The input question
        hops: Ordered list of reasoning hops
        final_answer: The final answer text
        supporting_facts: List of intermediate facts/nodes used as evidence
    """
    question: str
    hops: List[HopResult]
    final_answer: str
    supporting_facts: List[str] = field(default_factory=list)


@dataclass
class GoldMultihopAnnotation:
    """Gold-standard annotation for a multi-hop question.

    Attributes:
        question_id: Unique identifier for this question
        question: The question text
        gold_hops: Expected reasoning path (list of HopResults)
        gold_answer: Expected answer text
        gold_supporting_facts: Expected supporting facts/nodes
        question_type: Category of multi-hop question (e.g., "bridge", "comparison")
        num_hops: Number of hops required (e.g., 2, 3)
    """
    question_id: str
    question: str
    gold_hops: List[HopResult]
    gold_answer: str
    gold_supporting_facts: List[str] = field(default_factory=list)
    question_type: Optional[str] = None
    num_hops: Optional[int] = None


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation, normalize whitespace)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization after normalization."""
    return _normalize_text(text).split()


def hop_accuracy(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Calculate hop-level accuracy (fraction of hops that match gold path).

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        Fraction of hops that match [0, 1]. Returns 0 if path lengths differ.
    """
    pred_hops = predicted_path.hops
    gold_hops = gold_annotation.gold_hops

    if len(pred_hops) != len(gold_hops):
        return 0.0

    if not pred_hops:
        return 1.0  # Empty paths match

    correct = 0
    for pred_hop, gold_hop in zip(pred_hops, gold_hops):
        if (pred_hop.source_node == gold_hop.source_node and
            pred_hop.edge_type == gold_hop.edge_type and
            pred_hop.target_node == gold_hop.target_node):
            correct += 1

    return correct / len(pred_hops)


def path_exact_match(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Check if the entire reasoning path exactly matches gold path.

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_hops = predicted_path.hops
    gold_hops = gold_annotation.gold_hops

    if len(pred_hops) != len(gold_hops):
        return 0.0

    for pred_hop, gold_hop in zip(pred_hops, gold_hops):
        if (pred_hop.source_node != gold_hop.source_node or
            pred_hop.edge_type != gold_hop.edge_type or
            pred_hop.target_node != gold_hop.target_node):
            return 0.0

    return 1.0


def supporting_facts_precision(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Calculate precision for supporting facts (what fraction of predicted facts are correct).

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        Precision [0, 1]
    """
    pred_facts = set(predicted_path.supporting_facts)
    gold_facts = set(gold_annotation.gold_supporting_facts)

    if not pred_facts:
        return 0.0

    correct = len(pred_facts & gold_facts)
    return correct / len(pred_facts)


def supporting_facts_recall(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Calculate recall for supporting facts (what fraction of gold facts were retrieved).

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        Recall [0, 1]
    """
    pred_facts = set(predicted_path.supporting_facts)
    gold_facts = set(gold_annotation.gold_supporting_facts)

    if not gold_facts:
        return 1.0 if not pred_facts else 0.0

    correct = len(pred_facts & gold_facts)
    return correct / len(gold_facts)


def supporting_facts_f1(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Calculate F1 score for supporting facts.

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        F1 score [0, 1]
    """
    precision = supporting_facts_precision(predicted_path, gold_annotation)
    recall = supporting_facts_recall(predicted_path, gold_annotation)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def answer_exact_match(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Check if final answer exactly matches gold answer (normalized).

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_normalized = _normalize_text(predicted_path.final_answer)
    gold_normalized = _normalize_text(gold_annotation.gold_answer)
    return 1.0 if pred_normalized == gold_normalized else 0.0


def answer_f1(predicted_path: MultihopPath, gold_annotation: GoldMultihopAnnotation) -> float:
    """Calculate token-level F1 score for final answer.

    Args:
        predicted_path: Predicted reasoning path
        gold_annotation: Gold standard annotation

    Returns:
        F1 score [0, 1]
    """
    pred_tokens = Counter(_tokenize(predicted_path.final_answer))
    gold_tokens = Counter(_tokenize(gold_annotation.gold_answer))

    common = pred_tokens & gold_tokens
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / sum(pred_tokens.values()) if pred_tokens else 0.0
    recall = num_common / sum(gold_tokens.values()) if gold_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


@dataclass
class MultihopMetrics:
    """Container for computing multi-hop reasoning metrics across a dataset.

    Attributes:
        strict_hop_matching: If True, require hop indices to match exactly
    """
    strict_hop_matching: bool = True

    def compute_all(
        self,
        predicted_path: MultihopPath,
        gold_annotation: GoldMultihopAnnotation
    ) -> Dict[str, float]:
        """Compute all multi-hop metrics for a single example.

        Args:
            predicted_path: Predicted reasoning path
            gold_annotation: Gold standard annotation

        Returns:
            Dict of metric_name -> value
        """
        return {
            "hop_accuracy": hop_accuracy(predicted_path, gold_annotation),
            "path_exact_match": path_exact_match(predicted_path, gold_annotation),
            "supporting_facts_precision": supporting_facts_precision(predicted_path, gold_annotation),
            "supporting_facts_recall": supporting_facts_recall(predicted_path, gold_annotation),
            "supporting_facts_f1": supporting_facts_f1(predicted_path, gold_annotation),
            "answer_exact_match": answer_exact_match(predicted_path, gold_annotation),
            "answer_f1": answer_f1(predicted_path, gold_annotation),
        }

    def compute_by_question_type(
        self,
        predictions: List[MultihopPath],
        gold_annotations: List[GoldMultihopAnnotation]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics grouped by question type (e.g., bridge, comparison).

        Args:
            predictions: List of predicted paths
            gold_annotations: List of gold annotations (must match predictions 1:1)

        Returns:
            Dict mapping question_type -> metrics dict
        """
        if len(predictions) != len(gold_annotations):
            raise ValueError(f"Mismatched lengths: {len(predictions)} predictions vs {len(gold_annotations)} golds")

        # Group by question type
        by_type: Dict[str, Tuple[List[MultihopPath], List[GoldMultihopAnnotation]]] = {}
        for pred, gold in zip(predictions, gold_annotations):
            qtype = gold.question_type or "unknown"
            if qtype not in by_type:
                by_type[qtype] = ([], [])
            by_type[qtype][0].append(pred)
            by_type[qtype][1].append(gold)

        # Compute metrics for each type
        results = {}
        for qtype, (preds, golds) in by_type.items():
            all_metrics = [self.compute_all(p, g) for p, g in zip(preds, golds)]
            results[qtype] = self.aggregate(all_metrics)

        return results

    def compute_by_num_hops(
        self,
        predictions: List[MultihopPath],
        gold_annotations: List[GoldMultihopAnnotation]
    ) -> Dict[int, Dict[str, float]]:
        """Compute metrics grouped by number of hops required.

        Args:
            predictions: List of predicted paths
            gold_annotations: List of gold annotations (must match predictions 1:1)

        Returns:
            Dict mapping num_hops -> metrics dict
        """
        if len(predictions) != len(gold_annotations):
            raise ValueError(f"Mismatched lengths: {len(predictions)} predictions vs {len(gold_annotations)} golds")

        # Group by num_hops
        by_hops: Dict[int, Tuple[List[MultihopPath], List[GoldMultihopAnnotation]]] = {}
        for pred, gold in zip(predictions, gold_annotations):
            num_hops = gold.num_hops if gold.num_hops is not None else len(gold.gold_hops)
            if num_hops not in by_hops:
                by_hops[num_hops] = ([], [])
            by_hops[num_hops][0].append(pred)
            by_hops[num_hops][1].append(gold)

        # Compute metrics for each hop count
        results = {}
        for num_hops, (preds, golds) in by_hops.items():
            all_metrics = [self.compute_all(p, g) for p, g in zip(preds, golds)]
            results[num_hops] = self.aggregate(all_metrics)

        return results

    def aggregate(self, all_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple examples (mean).

        Args:
            all_results: List of metric dicts

        Returns:
            Dict with mean values for each metric
        """
        if not all_results:
            return {}

        metric_names = list(all_results[0].keys())
        aggregated = {}
        for name in metric_names:
            values = [r.get(name, 0.0) for r in all_results]
            aggregated[f"mean_{name}"] = sum(values) / len(values)

        return aggregated
