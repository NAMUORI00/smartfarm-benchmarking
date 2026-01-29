"""Evaluation metrics for RAG systems."""

from .retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    RetrievalMetrics,
)

from .qa_metrics import (
    exact_match,
    f1_score,
    bleu_score,
    rouge_score,
    semantic_similarity,
    QAMetrics,
)

from .domain_metrics import (
    ontology_coverage,
    term_accuracy,
    causal_chain_match,
    DomainMetrics,
)

from .extraction_metrics import (
    MatchMode,
    EntityMatch,
    RelationMatch,
    match_entities,
    match_relations,
    entity_precision,
    entity_recall,
    entity_f1,
    relation_precision,
    relation_recall,
    relation_f1,
    ExtractionMetrics,
)

from .multihop_metrics import (
    HopResult,
    MultihopPath,
    GoldMultihopAnnotation,
    hop_accuracy,
    path_exact_match,
    supporting_facts_precision,
    supporting_facts_recall,
    supporting_facts_f1,
    answer_exact_match,
    answer_f1,
    MultihopMetrics,
)

__all__ = [
    # Retrieval
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "RetrievalMetrics",
    # QA
    "exact_match",
    "f1_score",
    "bleu_score",
    "rouge_score",
    "semantic_similarity",
    "QAMetrics",
    # Domain
    "ontology_coverage",
    "term_accuracy",
    "causal_chain_match",
    "DomainMetrics",
    # Extraction
    "MatchMode",
    "EntityMatch",
    "RelationMatch",
    "match_entities",
    "match_relations",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "relation_precision",
    "relation_recall",
    "relation_f1",
    "ExtractionMetrics",
    # Multihop
    "HopResult",
    "MultihopPath",
    "GoldMultihopAnnotation",
    "hop_accuracy",
    "path_exact_match",
    "supporting_facts_precision",
    "supporting_facts_recall",
    "supporting_facts_f1",
    "answer_exact_match",
    "answer_f1",
    "MultihopMetrics",
]
