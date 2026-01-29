"""Entity and Relation extraction evaluation metrics.

Provides precision, recall, and F1 scores for causal knowledge extraction:
- Entity matching: exact, canonical ID, or fuzzy text matching
- Relation matching: source-target-type triples with configurable entity matching
- Type-specific metrics: per-entity-type and per-relation-type breakdowns
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas.CausalSchema import (
    CausalEntity,
    CausalRelation,
    EntityType,
    RelationType,
)


class MatchMode(str, Enum):
    """Entity matching mode for evaluation."""
    EXACT = "exact"  # Exact text match (normalized)
    CANONICAL = "canonical"  # Match by canonical_id
    FUZZY = "fuzzy"  # Fuzzy text similarity (>= threshold)


@dataclass
class EntityMatch:
    """Matched entity pair (predicted, gold)."""
    predicted: CausalEntity
    gold: CausalEntity
    match_score: float  # 1.0 for exact/canonical, similarity for fuzzy


@dataclass
class RelationMatch:
    """Matched relation pair (predicted, gold)."""
    predicted: CausalRelation
    gold: CausalRelation
    match_score: float


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def _fuzzy_similarity(text1: str, text2: str) -> float:
    """Compute fuzzy text similarity (Jaccard on character bigrams)."""
    def get_bigrams(s: str) -> Set[str]:
        s = _normalize_text(s)
        return set(s[i:i+2] for i in range(len(s) - 1))

    bigrams1 = get_bigrams(text1)
    bigrams2 = get_bigrams(text2)

    if not bigrams1 or not bigrams2:
        return 1.0 if text1 == text2 else 0.0

    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)

    return intersection / union if union > 0 else 0.0


def match_entities(
    predicted: List[CausalEntity],
    gold: List[CausalEntity],
    mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> List[EntityMatch]:
    """Match predicted entities to gold entities.

    Args:
        predicted: Predicted entities
        gold: Gold standard entities
        mode: Matching mode (exact, canonical, fuzzy)
        fuzzy_threshold: Minimum similarity for fuzzy matching

    Returns:
        List of matched entity pairs (predicted, gold)
    """
    matches: List[EntityMatch] = []
    used_gold_indices: Set[int] = set()

    for pred in predicted:
        best_match_idx: Optional[int] = None
        best_match_score: float = 0.0

        for i, g in enumerate(gold):
            if i in used_gold_indices:
                continue

            # Type must match
            if pred.entity_type != g.entity_type:
                continue

            score = 0.0

            if mode == MatchMode.EXACT:
                if _normalize_text(pred.text) == _normalize_text(g.text):
                    score = 1.0

            elif mode == MatchMode.CANONICAL:
                if pred.canonical_id and g.canonical_id:
                    if pred.canonical_id == g.canonical_id:
                        score = 1.0
                else:
                    # Fallback to exact text match if no canonical_id
                    if _normalize_text(pred.text) == _normalize_text(g.text):
                        score = 1.0

            elif mode == MatchMode.FUZZY:
                score = _fuzzy_similarity(pred.text, g.text)

            if score >= fuzzy_threshold and score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_idx is not None:
            matches.append(EntityMatch(
                predicted=pred,
                gold=gold[best_match_idx],
                match_score=best_match_score,
            ))
            used_gold_indices.add(best_match_idx)

    return matches


def _entity_matches(
    e1: CausalEntity,
    e2: CausalEntity,
    mode: MatchMode,
    fuzzy_threshold: float = 0.8,
) -> bool:
    """Check if two entities match according to mode."""
    if e1.entity_type != e2.entity_type:
        return False

    if mode == MatchMode.EXACT:
        return _normalize_text(e1.text) == _normalize_text(e2.text)

    elif mode == MatchMode.CANONICAL:
        if e1.canonical_id and e2.canonical_id:
            return e1.canonical_id == e2.canonical_id
        else:
            # Fallback to exact
            return _normalize_text(e1.text) == _normalize_text(e2.text)

    elif mode == MatchMode.FUZZY:
        return _fuzzy_similarity(e1.text, e2.text) >= fuzzy_threshold

    return False


def match_relations(
    predicted: List[CausalRelation],
    gold: List[CausalRelation],
    entity_mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> List[RelationMatch]:
    """Match predicted relations to gold relations.

    A relation matches if:
    1. Source entity matches (by entity_mode)
    2. Target entity matches (by entity_mode)
    3. Relation type matches

    Args:
        predicted: Predicted relations
        gold: Gold standard relations
        entity_mode: Matching mode for entities
        fuzzy_threshold: Minimum similarity for fuzzy entity matching

    Returns:
        List of matched relation pairs
    """
    matches: List[RelationMatch] = []
    used_gold_indices: Set[int] = set()

    for pred in predicted:
        for i, g in enumerate(gold):
            if i in used_gold_indices:
                continue

            # Relation type must match
            if pred.relation_type != g.relation_type:
                continue

            # Source and target must match
            source_match = _entity_matches(
                pred.source, g.source, entity_mode, fuzzy_threshold
            )
            target_match = _entity_matches(
                pred.target, g.target, entity_mode, fuzzy_threshold
            )

            if source_match and target_match:
                matches.append(RelationMatch(
                    predicted=pred,
                    gold=g,
                    match_score=1.0,  # Exact match (all components match)
                ))
                used_gold_indices.add(i)
                break  # Use first match

    return matches


def entity_precision(
    predicted: List[CausalEntity],
    gold: List[CausalEntity],
    mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> float:
    """Entity precision: correct predictions / total predictions.

    Args:
        predicted: Predicted entities
        gold: Gold standard entities
        mode: Matching mode
        fuzzy_threshold: Minimum similarity for fuzzy matching

    Returns:
        Precision [0, 1]
    """
    if not predicted:
        return 0.0

    matches = match_entities(predicted, gold, mode, fuzzy_threshold)
    return len(matches) / len(predicted)


def entity_recall(
    predicted: List[CausalEntity],
    gold: List[CausalEntity],
    mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> float:
    """Entity recall: correct predictions / total gold.

    Args:
        predicted: Predicted entities
        gold: Gold standard entities
        mode: Matching mode
        fuzzy_threshold: Minimum similarity for fuzzy matching

    Returns:
        Recall [0, 1]
    """
    if not gold:
        return 1.0 if not predicted else 0.0

    matches = match_entities(predicted, gold, mode, fuzzy_threshold)
    return len(matches) / len(gold)


def entity_f1(
    predicted: List[CausalEntity],
    gold: List[CausalEntity],
    mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> float:
    """Entity F1 score: harmonic mean of precision and recall.

    Args:
        predicted: Predicted entities
        gold: Gold standard entities
        mode: Matching mode
        fuzzy_threshold: Minimum similarity for fuzzy matching

    Returns:
        F1 score [0, 1]
    """
    p = entity_precision(predicted, gold, mode, fuzzy_threshold)
    r = entity_recall(predicted, gold, mode, fuzzy_threshold)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def relation_precision(
    predicted: List[CausalRelation],
    gold: List[CausalRelation],
    entity_mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> float:
    """Relation precision: correct predictions / total predictions.

    Args:
        predicted: Predicted relations
        gold: Gold standard relations
        entity_mode: Matching mode for entities
        fuzzy_threshold: Minimum similarity for fuzzy entity matching

    Returns:
        Precision [0, 1]
    """
    if not predicted:
        return 0.0

    matches = match_relations(predicted, gold, entity_mode, fuzzy_threshold)
    return len(matches) / len(predicted)


def relation_recall(
    predicted: List[CausalRelation],
    gold: List[CausalRelation],
    entity_mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> float:
    """Relation recall: correct predictions / total gold.

    Args:
        predicted: Predicted relations
        gold: Gold standard relations
        entity_mode: Matching mode for entities
        fuzzy_threshold: Minimum similarity for fuzzy entity matching

    Returns:
        Recall [0, 1]
    """
    if not gold:
        return 1.0 if not predicted else 0.0

    matches = match_relations(predicted, gold, entity_mode, fuzzy_threshold)
    return len(matches) / len(gold)


def relation_f1(
    predicted: List[CausalRelation],
    gold: List[CausalRelation],
    entity_mode: MatchMode = MatchMode.CANONICAL,
    fuzzy_threshold: float = 0.8,
) -> float:
    """Relation F1 score: harmonic mean of precision and recall.

    Args:
        predicted: Predicted relations
        gold: Gold standard relations
        entity_mode: Matching mode for entities
        fuzzy_threshold: Minimum similarity for fuzzy entity matching

    Returns:
        F1 score [0, 1]
    """
    p = relation_precision(predicted, gold, entity_mode, fuzzy_threshold)
    r = relation_recall(predicted, gold, entity_mode, fuzzy_threshold)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


@dataclass
class ExtractionMetrics:
    """Container for computing extraction evaluation metrics."""

    entity_mode: MatchMode = MatchMode.CANONICAL
    fuzzy_threshold: float = 0.8

    def compute_all(
        self,
        predicted_entities: List[CausalEntity],
        gold_entities: List[CausalEntity],
        predicted_relations: List[CausalRelation],
        gold_relations: List[CausalRelation],
    ) -> Dict[str, float]:
        """Compute all extraction metrics.

        Args:
            predicted_entities: Predicted entities
            gold_entities: Gold standard entities
            predicted_relations: Predicted relations
            gold_relations: Gold standard relations

        Returns:
            Dict of metric_name -> value
        """
        results = {
            "entity_precision": entity_precision(
                predicted_entities, gold_entities, self.entity_mode, self.fuzzy_threshold
            ),
            "entity_recall": entity_recall(
                predicted_entities, gold_entities, self.entity_mode, self.fuzzy_threshold
            ),
            "entity_f1": entity_f1(
                predicted_entities, gold_entities, self.entity_mode, self.fuzzy_threshold
            ),
            "relation_precision": relation_precision(
                predicted_relations, gold_relations, self.entity_mode, self.fuzzy_threshold
            ),
            "relation_recall": relation_recall(
                predicted_relations, gold_relations, self.entity_mode, self.fuzzy_threshold
            ),
            "relation_f1": relation_f1(
                predicted_relations, gold_relations, self.entity_mode, self.fuzzy_threshold
            ),
        }

        return results

    def compute_per_type(
        self,
        predicted_entities: List[CausalEntity],
        gold_entities: List[CausalEntity],
        predicted_relations: List[CausalRelation],
        gold_relations: List[CausalRelation],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by entity/relation type.

        Args:
            predicted_entities: Predicted entities
            gold_entities: Gold standard entities
            predicted_relations: Predicted relations
            gold_relations: Gold standard relations

        Returns:
            Dict with keys:
            - "entity_types": {type_name: {precision, recall, f1}}
            - "relation_types": {type_name: {precision, recall, f1}}
        """
        results: Dict[str, Dict[str, float]] = {
            "entity_types": {},
            "relation_types": {},
        }

        # Entity metrics per type
        for entity_type in EntityType:
            pred_filtered = [e for e in predicted_entities if e.entity_type == entity_type]
            gold_filtered = [e for e in gold_entities if e.entity_type == entity_type]

            if not pred_filtered and not gold_filtered:
                continue  # Skip types with no data

            results["entity_types"][entity_type.value] = {
                "precision": entity_precision(
                    pred_filtered, gold_filtered, self.entity_mode, self.fuzzy_threshold
                ),
                "recall": entity_recall(
                    pred_filtered, gold_filtered, self.entity_mode, self.fuzzy_threshold
                ),
                "f1": entity_f1(
                    pred_filtered, gold_filtered, self.entity_mode, self.fuzzy_threshold
                ),
            }

        # Relation metrics per type
        for relation_type in RelationType:
            pred_filtered = [r for r in predicted_relations if r.relation_type == relation_type]
            gold_filtered = [r for r in gold_relations if r.relation_type == relation_type]

            if not pred_filtered and not gold_filtered:
                continue  # Skip types with no data

            results["relation_types"][relation_type.value] = {
                "precision": relation_precision(
                    pred_filtered, gold_filtered, self.entity_mode, self.fuzzy_threshold
                ),
                "recall": relation_recall(
                    pred_filtered, gold_filtered, self.entity_mode, self.fuzzy_threshold
                ),
                "f1": relation_f1(
                    pred_filtered, gold_filtered, self.entity_mode, self.fuzzy_threshold
                ),
            }

        return results

    def aggregate(
        self,
        all_results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across multiple examples (mean).

        Args:
            all_results: List of metric dicts from compute_all()

        Returns:
            Aggregated metrics with "mean_" prefix
        """
        if not all_results:
            return {}

        metric_names = list(all_results[0].keys())
        aggregated = {}

        for name in metric_names:
            values = [r.get(name, 0.0) for r in all_results]
            aggregated[f"mean_{name}"] = sum(values) / len(values)

        return aggregated
