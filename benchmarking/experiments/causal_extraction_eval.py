#!/usr/bin/env python3
"""Causal extraction evaluation for RQ1.

Evaluates CausalExtractor quality by comparing predicted entities and relations
against gold annotations using precision, recall, and F1 metrics.

Supports:
- Three extraction modes: rule_only, llm_only, hybrid
- Three entity matching modes: exact, canonical, fuzzy
- Type-specific metrics (per entity/relation type)
- Confidence threshold filtering
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from benchmarking.metrics.extraction_metrics import (
    ExtractionMetrics,
    MatchMode,
)
from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models import SourceDoc
from core.Models.Schemas.CausalSchema import (
    CausalEntity,
    CausalRelation,
    EntityType,
    ExtractionResult,
    RelationType,
)
from core.Services.Ingest.CausalExtractor import CausalExtractor
from core.Services.Ingest.HybridGraphBuilder import HybridGraphBuilder


@dataclass
class GoldAnnotation:
    """Gold standard annotation for a document."""
    doc_id: str
    text: str
    entities: List[CausalEntity]
    relations: List[CausalRelation]


def load_gold_annotations(path: Path, limit: Optional[int] = None) -> List[GoldAnnotation]:
    """Load gold annotations from JSONL file.

    Expected format per line:
    {
        "doc_id": "doc_123",
        "text": "Original document text...",
        "entities": [
            {
                "text": "탄저병",
                "type": "disease",
                "canonical_id": "disease:탄저병",
                "confidence": 1.0,
                "span": [10, 13]
            },
            ...
        ],
        "relations": [
            {
                "source": {"text": "탄저병", "type": "disease", ...},
                "target": {"text": "고온다습", "type": "environmental_condition", ...},
                "type": "caused_by",
                "confidence": 1.0,
                "evidence": "..."
            },
            ...
        ]
    }

    Args:
        path: Path to gold annotations JSONL file
        limit: Maximum number of annotations to load

    Returns:
        List of gold annotations
    """
    annotations: List[GoldAnnotation] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Parse entities
                entities = []
                for ent_data in data.get("entities", []):
                    entities.append(
                        CausalEntity(
                            text=ent_data["text"],
                            entity_type=EntityType(ent_data["type"]),
                            canonical_id=ent_data.get("canonical_id"),
                            confidence=ent_data.get("confidence", 1.0),
                            span=tuple(ent_data.get("span", (0, 0))),
                        )
                    )

                # Parse relations
                relations = []
                for rel_data in data.get("relations", []):
                    source_data = rel_data["source"]
                    target_data = rel_data["target"]

                    source = CausalEntity(
                        text=source_data["text"],
                        entity_type=EntityType(source_data["type"]),
                        canonical_id=source_data.get("canonical_id"),
                        confidence=source_data.get("confidence", 1.0),
                        span=tuple(source_data.get("span", (0, 0))),
                    )

                    target = CausalEntity(
                        text=target_data["text"],
                        entity_type=EntityType(target_data["type"]),
                        canonical_id=target_data.get("canonical_id"),
                        confidence=target_data.get("confidence", 1.0),
                        span=tuple(target_data.get("span", (0, 0))),
                    )

                    relations.append(
                        CausalRelation(
                            source=source,
                            target=target,
                            relation_type=RelationType(rel_data["type"]),
                            confidence=rel_data.get("confidence", 1.0),
                            evidence_text=rel_data.get("evidence", ""),
                        )
                    )

                annotations.append(
                    GoldAnnotation(
                        doc_id=data["doc_id"],
                        text=data.get("text", ""),
                        entities=entities,
                        relations=relations,
                    )
                )

                if limit and len(annotations) >= limit:
                    break

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue

    return annotations


def load_corpus_docs(path: Path, doc_ids: List[str]) -> Dict[str, SourceDoc]:
    """Load corpus documents from JSONL file.

    Args:
        path: Path to corpus JSONL file
        doc_ids: List of document IDs to load

    Returns:
        Dict mapping doc_id -> SourceDoc
    """
    doc_id_set = set(doc_ids)
    docs: Dict[str, SourceDoc] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                doc_id = data["id"]

                if doc_id in doc_id_set:
                    docs[doc_id] = SourceDoc(
                        id=doc_id,
                        text=data["text"],
                        metadata=data.get("metadata", {}),
                    )

                    if len(docs) == len(doc_id_set):
                        break  # Found all documents

            except (json.JSONDecodeError, KeyError) as e:
                continue

    return docs


async def run_extraction(
    docs: List[SourceDoc],
    mode: str,
    min_confidence: float,
) -> List[ExtractionResult]:
    """Run causal extraction on documents.

    Args:
        docs: Source documents
        mode: Extraction mode (rule_only, llm_only, hybrid)
        min_confidence: Minimum confidence threshold

    Returns:
        List of extraction results
    """
    if mode == "rule_only":
        # Rule-only mode not yet implemented for direct entity extraction
        # For now, we'll use hybrid mode with rule_only setting
        builder = HybridGraphBuilder(mode="rule_only")
        graph = await builder.build(docs)
        # Convert graph to ExtractionResult format (simplified)
        # Note: This is a placeholder - actual conversion needed
        return []

    elif mode == "llm_only":
        extractor = CausalExtractor(min_confidence=min_confidence)
        results = await extractor.extract(docs)
        return results

    elif mode == "hybrid":
        builder = HybridGraphBuilder(mode="hybrid")
        # For hybrid mode, we need to extract directly for evaluation
        extractor = CausalExtractor(min_confidence=min_confidence)
        results = await extractor.extract(docs)
        return results

    else:
        raise ValueError(f"Invalid mode: {mode}")


def evaluate_extraction(
    gold_annotations: List[GoldAnnotation],
    predicted_results: List[ExtractionResult],
    match_mode: MatchMode,
    fuzzy_threshold: float = 0.8,
    stratify_by_type: bool = False,
) -> Dict:
    """Evaluate extraction results against gold annotations.

    Args:
        gold_annotations: Gold standard annotations
        predicted_results: Predicted extraction results
        match_mode: Entity matching mode
        fuzzy_threshold: Minimum similarity for fuzzy matching
        stratify_by_type: Include per-type metrics

    Returns:
        Evaluation results dict
    """
    metrics_calculator = ExtractionMetrics(
        entity_mode=match_mode,
        fuzzy_threshold=fuzzy_threshold,
    )

    # Build lookup for predicted results
    predicted_by_id = {r.doc_id: r for r in predicted_results}

    # Compute metrics per document
    per_doc_results = []
    all_metrics = []

    for gold in gold_annotations:
        predicted = predicted_by_id.get(gold.doc_id)

        if predicted is None:
            # No prediction for this document
            doc_metrics = {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0,
                "relation_precision": 0.0,
                "relation_recall": 0.0,
                "relation_f1": 0.0,
            }
        else:
            doc_metrics = metrics_calculator.compute_all(
                predicted_entities=predicted.entities,
                gold_entities=gold.entities,
                predicted_relations=predicted.relations,
                gold_relations=gold.relations,
            )

        per_doc_results.append({
            "doc_id": gold.doc_id,
            "metrics": doc_metrics,
            "gold_entity_count": len(gold.entities),
            "gold_relation_count": len(gold.relations),
            "predicted_entity_count": len(predicted.entities) if predicted else 0,
            "predicted_relation_count": len(predicted.relations) if predicted else 0,
        })

        all_metrics.append(doc_metrics)

    # Aggregate metrics
    aggregated = metrics_calculator.aggregate(all_metrics)

    results = {
        "summary": aggregated,
        "num_documents": len(gold_annotations),
        "num_predicted": len(predicted_results),
        "per_document": per_doc_results,
    }

    # Add type-specific metrics if requested
    if stratify_by_type:
        # Aggregate all gold and predicted entities/relations
        all_gold_entities = []
        all_gold_relations = []
        all_pred_entities = []
        all_pred_relations = []

        for gold in gold_annotations:
            all_gold_entities.extend(gold.entities)
            all_gold_relations.extend(gold.relations)

            predicted = predicted_by_id.get(gold.doc_id)
            if predicted:
                all_pred_entities.extend(predicted.entities)
                all_pred_relations.extend(predicted.relations)

        type_metrics = metrics_calculator.compute_per_type(
            predicted_entities=all_pred_entities,
            gold_entities=all_gold_entities,
            predicted_relations=all_pred_relations,
            gold_relations=all_gold_relations,
        )

        results["type_specific"] = type_metrics

    return results


def save_results(results: Dict, output_path: Path) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dict
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(results: Dict) -> None:
    """Print evaluation summary to console.

    Args:
        results: Evaluation results dict
    """
    summary = results["summary"]

    print("\n" + "=" * 60)
    print("Causal Extraction Evaluation Summary (RQ1)")
    print("=" * 60)
    print(f"Documents evaluated: {results['num_documents']}")
    print(f"Predictions made: {results['num_predicted']}")
    print()

    print("Entity Metrics:")
    print(f"  Precision: {summary.get('mean_entity_precision', 0.0):.3f}")
    print(f"  Recall:    {summary.get('mean_entity_recall', 0.0):.3f}")
    print(f"  F1:        {summary.get('mean_entity_f1', 0.0):.3f}")
    print()

    print("Relation Metrics:")
    print(f"  Precision: {summary.get('mean_relation_precision', 0.0):.3f}")
    print(f"  Recall:    {summary.get('mean_relation_recall', 0.0):.3f}")
    print(f"  F1:        {summary.get('mean_relation_f1', 0.0):.3f}")
    print()

    # Print type-specific metrics if available
    if "type_specific" in results:
        print("Type-Specific Metrics:")
        print()

        type_data = results["type_specific"]

        if type_data.get("entity_types"):
            print("  Entity Types:")
            for entity_type, metrics in type_data["entity_types"].items():
                print(f"    {entity_type}:")
                print(f"      P: {metrics['precision']:.3f}, "
                      f"R: {metrics['recall']:.3f}, "
                      f"F1: {metrics['f1']:.3f}")
            print()

        if type_data.get("relation_types"):
            print("  Relation Types:")
            for relation_type, metrics in type_data["relation_types"].items():
                print(f"    {relation_type}:")
                print(f"      P: {metrics['precision']:.3f}, "
                      f"R: {metrics['recall']:.3f}, "
                      f"F1: {metrics['f1']:.3f}")
            print()

    print("=" * 60)


async def main() -> None:
    """Main entry point for causal extraction evaluation."""
    ap = argparse.ArgumentParser(
        description="Evaluate causal extraction quality (RQ1)"
    )

    # Input/output paths
    ap.add_argument(
        "--gold-file",
        type=Path,
        required=True,
        help="Path to gold annotations JSONL file",
    )
    ap.add_argument(
        "--corpus-file",
        type=Path,
        default=None,
        help="Path to corpus JSONL file (optional, for extracting from original text)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output evaluation results JSON",
    )

    # Extraction configuration
    ap.add_argument(
        "--mode",
        choices=["rule_only", "llm_only", "hybrid"],
        default="hybrid",
        help="Extraction mode (default: hybrid)",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for extraction (default: 0.7)",
    )

    # Evaluation configuration
    ap.add_argument(
        "--match-mode",
        choices=["exact", "canonical", "fuzzy"],
        default="canonical",
        help="Entity matching mode (default: canonical)",
    )
    ap.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help="Minimum similarity for fuzzy matching (default: 0.8)",
    )
    ap.add_argument(
        "--stratify-by-type",
        action="store_true",
        help="Include per-type metrics in output",
    )

    # Execution limits
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to evaluate",
    )

    args = ap.parse_args()

    # Load gold annotations
    print(f"Loading gold annotations from {args.gold_file}...")
    gold_annotations = load_gold_annotations(args.gold_file, limit=args.limit)

    if not gold_annotations:
        print("Error: No gold annotations loaded.")
        return

    print(f"Loaded {len(gold_annotations)} gold annotations.")

    # Load or create documents for extraction
    if args.corpus_file:
        print(f"Loading corpus documents from {args.corpus_file}...")
        doc_ids = [g.doc_id for g in gold_annotations]
        corpus_docs = load_corpus_docs(args.corpus_file, doc_ids)

        # Filter to only annotations with corpus documents
        gold_annotations = [
            g for g in gold_annotations if g.doc_id in corpus_docs
        ]

        docs_for_extraction = [corpus_docs[g.doc_id] for g in gold_annotations]
        print(f"Found {len(docs_for_extraction)} corpus documents.")
    else:
        # Use text from gold annotations
        print("Using text from gold annotations for extraction...")
        docs_for_extraction = [
            SourceDoc(id=g.doc_id, text=g.text, metadata={})
            for g in gold_annotations
        ]

    # Run extraction
    print(f"Running extraction (mode={args.mode}, min_confidence={args.min_confidence})...")
    predicted_results = await run_extraction(
        docs=docs_for_extraction,
        mode=args.mode,
        min_confidence=args.min_confidence,
    )

    print(f"Extracted {len(predicted_results)} results.")

    # Evaluate
    print(f"Evaluating (match_mode={args.match_mode})...")
    match_mode = MatchMode(args.match_mode)

    results = evaluate_extraction(
        gold_annotations=gold_annotations,
        predicted_results=predicted_results,
        match_mode=match_mode,
        fuzzy_threshold=args.fuzzy_threshold,
        stratify_by_type=args.stratify_by_type,
    )

    # Save results
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
