#!/usr/bin/env python3
"""HybridDAT + PathRAG Benchmark

Compares HybridDATRetriever with three PathRAG configurations:
1. use_pathrag=False (Dense+Sparse only)
2. use_pathrag=True with OLD PathRAGRetriever
3. use_pathrag=True with NEW PathRAGLtRetriever

Usage:
    python -m benchmarking.experiments.hybrid_pathrag_lt_benchmark

Metrics:
    - MRR (Mean Reciprocal Rank)
    - Precision@K (K=1,4,5,10)
    - Recall@K
    - Hit Rate@K
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Config.Settings import settings
from core.Models.Schemas import SourceDoc
from core.Services.Ingest.GraphBuilder import build_graph_from_docs
from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store
from core.Services.Retrieval.Hybrid import HybridDATRetriever
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
    latencies = []

    total_queries = len(qa_items)
    print(f"\n{name}: Evaluating on {total_queries} queries...")

    for idx, qa in enumerate(qa_items, 1):
        if idx % 10 == 0:
            sys.stdout.write(f"\r  Progress: {idx}/{total_queries}")
            sys.stdout.flush()

        # Retrieve documents with timing
        start = time.perf_counter()
        results = retriever.search(qa.question, k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

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
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

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
        "latency_ms": avg_latency,
    }


def print_comparison(no_pathrag: dict, old_pathrag: dict, new_pathrag: dict):
    """Print comparison table (ASCII-safe for Windows)."""
    print("\n" + "=" * 95)
    print("HybridDAT + PathRAG Benchmark: Three Configurations")
    print("=" * 95)

    print(f"\n{'Metric':<20} {'No PathRAG':>15} {'Old PathRAG':>15} {'New PathRAGLt':>15} {'Old Improv':>15} {'New Improv':>15}")
    print("-" * 95)

    metrics = [
        "mrr",
        "precision@1", "precision@4", "precision@5", "precision@10",
        "recall@1", "recall@4", "recall@5", "recall@10",
        "hit_rate@1", "hit_rate@4", "hit_rate@5", "hit_rate@10",
        "latency_ms",
    ]

    for metric in metrics:
        no_val = no_pathrag.get(metric, 0.0)
        old_val = old_pathrag.get(metric, 0.0)
        new_val = new_pathrag.get(metric, 0.0)

        # Calculate improvements vs no_pathrag baseline
        old_improvement = ((old_val - no_val) / no_val * 100) if no_val > 0 else 0.0
        new_improvement = ((new_val - no_val) / no_val * 100) if no_val > 0 else 0.0

        old_imp_str = f"+{old_improvement:.1f}%" if old_improvement > 0 else f"{old_improvement:.1f}%"
        new_imp_str = f"+{new_improvement:.1f}%" if new_improvement > 0 else f"{new_improvement:.1f}%"

        # Format metric value
        if metric == "latency_ms":
            print(f"{metric:<20} {no_val:>15.2f} {old_val:>15.2f} {new_val:>15.2f} {old_imp_str:>15} {new_imp_str:>15}")
        else:
            print(f"{metric:<20} {no_val:>15.4f} {old_val:>15.4f} {new_val:>15.4f} {old_imp_str:>15} {new_imp_str:>15}")

    print("=" * 95)

    # Summary
    print(f"\nKey Improvements (vs No PathRAG baseline):")
    print(f"\n  MRR:")
    print(f"    No PathRAG:      {no_pathrag['mrr']:.4f}")
    print(f"    Old PathRAG:     {old_pathrag['mrr']:.4f} ({((old_pathrag['mrr'] - no_pathrag['mrr']) / no_pathrag['mrr'] * 100):+.1f}%)")
    print(f"    New PathRAGLt:   {new_pathrag['mrr']:.4f} ({((new_pathrag['mrr'] - no_pathrag['mrr']) / no_pathrag['mrr'] * 100):+.1f}%)")

    print(f"\n  Precision@4:")
    print(f"    No PathRAG:      {no_pathrag['precision@4']:.4f}")
    print(f"    Old PathRAG:     {old_pathrag['precision@4']:.4f} ({((old_pathrag['precision@4'] - no_pathrag['precision@4']) / no_pathrag['precision@4'] * 100):+.1f}%)")
    print(f"    New PathRAGLt:   {new_pathrag['precision@4']:.4f} ({((new_pathrag['precision@4'] - no_pathrag['precision@4']) / no_pathrag['precision@4'] * 100):+.1f}%)")

    print(f"\n  Recall@4:")
    print(f"    No PathRAG:      {no_pathrag['recall@4']:.4f}")
    print(f"    Old PathRAG:     {old_pathrag['recall@4']:.4f} ({((old_pathrag['recall@4'] - no_pathrag['recall@4']) / no_pathrag['recall@4'] * 100):+.1f}%)")
    print(f"    New PathRAGLt:   {new_pathrag['recall@4']:.4f} ({((new_pathrag['recall@4'] - no_pathrag['recall@4']) / no_pathrag['recall@4'] * 100):+.1f}%)")

    print(f"\n  Latency:")
    print(f"    No PathRAG:      {no_pathrag['latency_ms']:.2f}ms")
    print(f"    Old PathRAG:     {old_pathrag['latency_ms']:.2f}ms ({((old_pathrag['latency_ms'] - no_pathrag['latency_ms']) / no_pathrag['latency_ms'] * 100):+.1f}%)")
    print(f"    New PathRAGLt:   {new_pathrag['latency_ms']:.2f}ms ({((new_pathrag['latency_ms'] - no_pathrag['latency_ms']) / no_pathrag['latency_ms'] * 100):+.1f}%)")


def main():
    """Run HybridDAT + PathRAG benchmark."""
    # Resolve paths from Settings
    script_dir = Path(__file__).resolve().parent
    era_rag_dir = script_dir.parent.parent

    corpus_path = Path(settings.get_corpus_path())
    qa_path = Path(settings.get_qa_dataset_path())

    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)

    if not qa_path.exists():
        print(f"Error: QA dataset not found: {qa_path}")
        sys.exit(1)

    print("=" * 95)
    print("HybridDAT + PathRAG Benchmark: Comparing Three Configurations")
    print("=" * 95)

    # Load data
    print("\n[1/5] Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"  Loaded {len(corpus)} documents")

    print("\n[2/5] Loading QA dataset...")
    qa_items = load_qa_dataset(qa_path)
    print(f"  Loaded {len(qa_items)} QA pairs")

    # Build shared indices
    print("\n[3/5] Building shared indices...")
    print("  - Building Dense index...")
    dense_retriever = EmbeddingRetriever()
    dense_retriever.build(corpus)
    print(f"    Dense index ready: {len(corpus)} documents")

    print("  - Building Sparse index (BM25)...")
    sparse_store = BM25Store()
    sparse_store.index(corpus)
    print(f"    Sparse index ready: {len(corpus)} documents")

    print("  - Building graph for PathRAG...")
    graph = build_graph_from_docs(corpus)
    print(f"    Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Create PathRAG retrievers
    old_pathrag = PathRAGRetriever(graph)
    new_pathrag = PathRAGLtRetriever(graph, max_hops=3, threshold=0.3, alpha=0.8)
    print("  PathRAG retrievers ready")

    # Create HybridDAT configurations
    print("\n[4/5] Creating HybridDAT configurations...")

    # Config 1: No PathRAG (Dense + Sparse only)
    config_no_pathrag = HybridDATRetriever(
        dense=dense_retriever,
        sparse=sparse_store,
        pathrag=None,
        use_rrf=False,
        use_dat=True,
        use_ontology=True,
        use_pathrag=False,  # Disable PathRAG
    )
    print("  - Config 1: No PathRAG (Dense+Sparse only)")

    # Config 2: Old PathRAG
    config_old_pathrag = HybridDATRetriever(
        dense=dense_retriever,
        sparse=sparse_store,
        pathrag=old_pathrag,
        use_rrf=False,
        use_dat=True,
        use_ontology=True,
        use_pathrag=True,  # Enable PathRAG
    )
    print("  - Config 2: Old PathRAG (Dense+Sparse+Old PathRAG)")

    # Config 3: New PathRAGLt
    config_new_pathrag = HybridDATRetriever(
        dense=dense_retriever,
        sparse=sparse_store,
        pathrag=new_pathrag,
        use_rrf=False,
        use_dat=True,
        use_ontology=True,
        use_pathrag=True,  # Enable PathRAG
    )
    print("  - Config 3: New PathRAGLt (Dense+Sparse+New PathRAGLt)")

    # Evaluate
    print("\n[5/5] Running benchmark...")
    no_pathrag_results = evaluate_retriever(config_no_pathrag, qa_items, k=10, name="Config 1: No PathRAG")
    old_pathrag_results = evaluate_retriever(config_old_pathrag, qa_items, k=10, name="Config 2: Old PathRAG")
    new_pathrag_results = evaluate_retriever(config_new_pathrag, qa_items, k=10, name="Config 3: New PathRAGLt")

    # Print comparison
    print_comparison(no_pathrag_results, old_pathrag_results, new_pathrag_results)

    # Save results
    output_dir = era_rag_dir / "benchmarking" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "hybrid_pathrag_benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "no_pathrag": no_pathrag_results,
            "old_pathrag": old_pathrag_results,
            "new_pathrag_lt": new_pathrag_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
