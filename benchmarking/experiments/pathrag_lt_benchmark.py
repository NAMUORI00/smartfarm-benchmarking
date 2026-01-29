#!/usr/bin/env python3
"""PathRAG vs PathRAGLt Benchmark

Compares old PathRAGRetriever vs new PathRAGLtRetriever on wasabi QA dataset.

Usage:
    python -m benchmarking.experiments.pathrag_lt_benchmark

Metrics:
    - MRR (Mean Reciprocal Rank)
    - Precision@K (K=1,4,5,10)
    - Recall@K
    - Hit Rate@K
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Ingest.GraphBuilder import build_graph_from_docs
from core.Services.Retrieval.PathRAG import PathRAGRetriever, PathRAGLtRetriever
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


def evaluate_retriever(retriever, qa_items: List[QAItem], k: int = 5, name: str = "Retriever") -> dict:
    """Evaluate a retriever on QA dataset."""
    mrr_scores = []
    precision_scores = {1: [], 4: [], 5: [], 10: []}
    recall_scores = {1: [], 4: [], 5: [], 10: []}
    hit_rate_scores = {1: [], 4: [], 5: [], 10: []}

    total_queries = len(qa_items)
    print(f"\n{name}: Evaluating on {total_queries} queries...")

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

        for k_val in [1, 4, 5, 10]:
            precision_scores[k_val].append(precision_at_k(retrieved_ids, relevant_ids, k_val))
            recall_scores[k_val].append(recall_at_k(retrieved_ids, relevant_ids, k_val))
            hit_rate_scores[k_val].append(hit_rate(retrieved_ids, relevant_ids, k_val))

    sys.stdout.write(f"\r  Progress: {total_queries}/{total_queries}\n")
    sys.stdout.flush()

    # Calculate averages
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_precision = {k_val: sum(scores) / len(scores) if scores else 0.0
                     for k_val, scores in precision_scores.items()}
    avg_recall = {k_val: sum(scores) / len(scores) if scores else 0.0
                  for k_val, scores in recall_scores.items()}
    avg_hit_rate = {k_val: sum(scores) / len(scores) if scores else 0.0
                    for k_val, scores in hit_rate_scores.items()}

    return {
        "mrr": avg_mrr,
        "precision@1": avg_precision[1],
        "precision@4": avg_precision[4],
        "precision@5": avg_precision[5],
        "precision@10": avg_precision[10],
        "recall@1": avg_recall[1],
        "recall@4": avg_recall[4],
        "recall@5": avg_recall[5],
        "recall@10": avg_recall[10],
        "hit_rate@1": avg_hit_rate[1],
        "hit_rate@4": avg_hit_rate[4],
        "hit_rate@5": avg_hit_rate[5],
        "hit_rate@10": avg_hit_rate[10],
    }


def print_comparison(old_results: dict, new_results: dict):
    """Print comparison table (ASCII-safe for Windows)."""
    print("\n" + "=" * 80)
    print("PathRAG Benchmark: Old vs New")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Old PathRAG':>15} {'New PathRAGLt':>15} {'Improvement':>15}")
    print("-" * 80)

    metrics = [
        "mrr",
        "precision@1", "precision@4", "precision@5", "precision@10",
        "recall@1", "recall@4", "recall@5", "recall@10",
        "hit_rate@1", "hit_rate@4", "hit_rate@5", "hit_rate@10",
    ]

    for metric in metrics:
        old_val = old_results.get(metric, 0.0)
        new_val = new_results.get(metric, 0.0)
        improvement = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0.0

        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        print(f"{metric:<20} {old_val:>15.4f} {new_val:>15.4f} {improvement_str:>15}")

    print("=" * 80)

    # Summary
    mrr_improvement = ((new_results["mrr"] - old_results["mrr"]) / old_results["mrr"] * 100) if old_results["mrr"] > 0 else 0.0
    p4_improvement = ((new_results["precision@4"] - old_results["precision@4"]) / old_results["precision@4"] * 100) if old_results["precision@4"] > 0 else 0.0

    print(f"\nKey Improvements:")
    print(f"  MRR:          {old_results['mrr']:.4f} -> {new_results['mrr']:.4f} ({mrr_improvement:+.1f}%)")
    print(f"  Precision@4:  {old_results['precision@4']:.4f} -> {new_results['precision@4']:.4f} ({p4_improvement:+.1f}%)")


def main():
    """Run PathRAG benchmark."""
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

    print("=" * 80)
    print("PathRAG Benchmark: Comparing Old vs New Implementation")
    print("=" * 80)

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

    # Create retrievers
    old_retriever = PathRAGRetriever(graph)
    new_retriever = PathRAGLtRetriever(graph, max_hops=3, threshold=0.3, alpha=0.8)

    # Evaluate
    print("\n[4/4] Running benchmark...")
    old_results = evaluate_retriever(old_retriever, qa_items, k=10, name="Old PathRAG")
    new_results = evaluate_retriever(new_retriever, qa_items, k=10, name="New PathRAGLt")

    # Print comparison
    print_comparison(old_results, new_results)

    # Save results
    output_dir = era_rag_dir / "benchmarking" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "pathrag_benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "old_pathrag": old_results,
            "new_pathrag_lt": new_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
