#!/usr/bin/env python3
"""Seed Strategy Benchmark for PathRAGLtRetriever

Tests all 4 seed matching strategies on the actual QA dataset:
- ontology: Strict ontology matching (original)
- keyword: Flexible keyword matching (recommended)
- metadata: Metadata-based matching
- all: Use all concept nodes (broadest coverage)

Usage:
    python -m benchmarking.experiments.seed_strategy_benchmark

Metrics:
    - Hit Rate: % of queries with at least 1 result (coverage indicator)
    - MRR (Mean Reciprocal Rank)
    - Precision@4
    - Recall@4
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Ingest.GraphBuilder import build_graph_from_docs
from core.Services.Retrieval.PathRAG import PathRAGLtRetriever
from benchmarking.metrics.retrieval_metrics import mrr, precision_at_k, recall_at_k, hit_rate


@dataclass
class QAItem:
    """QA dataset item."""
    id: str
    question: str
    answer: str
    context: str
    category: str
    complexity: str
    source_ids: List[str] = field(default_factory=list)


def load_corpus(corpus_path: Path) -> List[SourceDoc]:
    """Load corpus from JSONL file."""
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc = SourceDoc(
                id=data["id"],
                text=data.get("text_ko", data.get("text_en", "")),
                metadata=data.get("metadata", {}),
            )
            docs.append(doc)
    return docs


def load_qa_dataset(qa_path: Path) -> List[QAItem]:
    """Load QA dataset from JSONL file."""
    qa_items = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            qa = QAItem(
                id=data["id"],
                question=data["question"],
                answer=data["answer"],
                context=data.get("context", ""),
                category=data.get("category", "unknown"),
                complexity=data.get("complexity", "unknown"),
                source_ids=data.get("source_ids", []),
            )
            qa_items.append(qa)
    return qa_items


def evaluate_seed_strategy(
    graph,
    qa_items: List[QAItem],
    seed_mode: str,
    k: int = 5,
) -> dict:
    """Evaluate PathRAGLt with specific seed matching strategy."""
    retriever = PathRAGLtRetriever(
        graph,
        max_hops=3,
        threshold=0.3,
        alpha=0.8,
        seed_match_mode=seed_mode,
    )

    mrr_scores = []
    precision_scores = []
    recall_scores = []
    hit_rate_scores = []

    total_queries = len(qa_items)
    print(f"\n{seed_mode.upper()}: Evaluating on {total_queries} queries...")

    for idx, qa in enumerate(qa_items, 1):
        if idx % 10 == 0:
            sys.stdout.write(f"\r  Progress: {idx}/{total_queries}")
            sys.stdout.flush()

        # Retrieve documents
        results = retriever.search(qa.question, k=k)
        retrieved_ids = [doc.id for doc in results]
        relevant_ids = set(qa.source_ids)

        # Calculate metrics
        mrr_scores.append(mrr(retrieved_ids, relevant_ids))
        precision_scores.append(precision_at_k(retrieved_ids, relevant_ids, k=4))
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=4))
        hit_rate_scores.append(hit_rate(retrieved_ids, relevant_ids, k=4))

    sys.stdout.write(f"\r  Progress: {total_queries}/{total_queries}\n")
    sys.stdout.flush()

    # Calculate averages
    return {
        "seed_mode": seed_mode,
        "hit_rate": sum(hit_rate_scores) / len(hit_rate_scores) if hit_rate_scores else 0.0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "precision@4": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        "recall@4": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
    }


def print_comparison_table(results_list: List[dict]):
    """Print ASCII comparison table for Windows compatibility."""
    print("\n" + "=" * 90)
    print("Seed Strategy Benchmark: Coverage & Retrieval Quality Comparison")
    print("=" * 90)

    print(f"\n{'Strategy':<15} {'Hit Rate':>12} {'MRR':>12} {'Precision@4':>15} {'Recall@4':>12}")
    print("-" * 90)

    for result in results_list:
        mode = result["seed_mode"]
        hit = result["hit_rate"]
        mrr_val = result["mrr"]
        prec = result["precision@4"]
        rec = result["recall@4"]

        print(f"{mode:<15} {hit:>12.4f} {mrr_val:>12.4f} {prec:>15.4f} {rec:>12.4f}")

    print("=" * 90)

    # Find best strategy for each metric
    best_hit = max(results_list, key=lambda x: x["hit_rate"])
    best_mrr = max(results_list, key=lambda x: x["mrr"])
    best_prec = max(results_list, key=lambda x: x["precision@4"])

    print("\nBest Strategies:")
    print(f"  Hit Rate (Coverage):    {best_hit['seed_mode']:>10} ({best_hit['hit_rate']:.4f})")
    print(f"  MRR:                    {best_mrr['seed_mode']:>10} ({best_mrr['mrr']:.4f})")
    print(f"  Precision@4 (Quality):  {best_prec['seed_mode']:>10} ({best_prec['precision@4']:.4f})")

    print("\nInterpretation:")
    print("  - Hit Rate: Higher = better coverage (more queries return results)")
    print("  - MRR: Higher = relevant docs ranked earlier")
    print("  - Precision@4: Higher = better quality of top-4 results")
    print("  - Recall@4: Higher = more relevant docs retrieved in top-4")


def main():
    """Run seed strategy benchmark."""
    # Resolve paths from script location
    script_dir = Path(__file__).resolve().parent
    # script_dir = .../era-smartfarm-rag/benchmarking/experiments
    # era_rag_dir = .../era-smartfarm-rag
    # workspace_dir = .../smartfarm-workspace-1
    era_rag_dir = script_dir.parent.parent
    workspace_dir = era_rag_dir.parent

    corpus_path = workspace_dir / "smartfarm-ingest" / "output" / "wasabi_en_ko_parallel.jsonl"
    qa_path = workspace_dir / "smartfarm-ingest" / "output" / "wasabi_qa_dataset.jsonl"

    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)

    if not qa_path.exists():
        print(f"Error: QA dataset not found: {qa_path}")
        sys.exit(1)

    print("=" * 90)
    print("Seed Strategy Benchmark: Testing 4 Seed Matching Modes")
    print("=" * 90)

    # Load data
    print("\n[1/4] Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"  Loaded {len(corpus)} documents")

    print("\n[2/4] Loading QA dataset...")
    qa_items = load_qa_dataset(qa_path)
    print(f"  Loaded {len(qa_items)} QA pairs")

    print("\n[3/4] Building graph from corpus...")
    graph = build_graph_from_docs(corpus)
    print(f"  Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Test all 4 strategies
    print("\n[4/4] Testing all seed strategies...")
    strategies = ["ontology", "keyword", "metadata", "all"]
    results_list = []

    for strategy in strategies:
        result = evaluate_seed_strategy(graph, qa_items, strategy, k=5)
        results_list.append(result)

    # Print comparison
    print_comparison_table(results_list)

    # Save results
    output_dir = era_rag_dir / "benchmarking" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "seed_strategy_benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "strategies": results_list,
            "graph_stats": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
            "dataset_stats": {
                "corpus_size": len(corpus),
                "qa_pairs": len(qa_items),
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
