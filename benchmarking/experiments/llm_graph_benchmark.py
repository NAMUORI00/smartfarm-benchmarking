#!/usr/bin/env python3
"""LLM-based Causal Extraction Graph Benchmark

Evaluates three graph building approaches:
1. Rule-only: Traditional pattern matching (GraphBuilder)
2. LLM-only: LLM-based causal extraction (CausalExtractor + LLMGraphBuilder)
3. Hybrid: Combined rule-based + LLM extraction (HybridGraphBuilder)

Usage:
    python -m benchmarking.experiments.llm_graph_benchmark
    python -m benchmarking.experiments.llm_graph_benchmark --modes rule_only llm_only
    python -m benchmarking.experiments.llm_graph_benchmark --retrieval-k 5

Metrics:
    - MRR (Mean Reciprocal Rank)
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Precision@K (K=5)
    - Recall@K (K=10)
    - Latency statistics (mean, p50, p95)
    - Graph statistics (nodes, edges, edge types)
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import statistics

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Config.Settings import settings
from core.Models.Schemas import SourceDoc
from core.Models.Graph import SmartFarmGraph
from core.Services.Ingest.GraphBuilder import build_graph_from_docs
from core.Services.Ingest.HybridGraphBuilder import HybridGraphBuilder
from core.Services.Retrieval.PathRAG import PathRAGLtRetriever
from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from benchmarking.metrics.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


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


def compute_graph_stats(graph: SmartFarmGraph) -> Dict[str, Any]:
    """Compute graph statistics.

    Args:
        graph: Knowledge graph

    Returns:
        Dict with node count, edge count, and edge type distribution
    """
    edge_types = {}
    for edge in graph.edges:
        edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

    return {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "edge_types": edge_types,
    }


def evaluate_retriever(
    retriever: PathRAGLtRetriever,
    qa_items: List[QAItem],
    k: int = 10,
    name: str = "Retriever",
) -> Dict[str, Any]:
    """Evaluate retriever on QA dataset.

    Args:
        retriever: PathRAG retriever instance
        qa_items: QA dataset items
        k: Top-k for retrieval
        name: Display name for progress

    Returns:
        Dict of evaluation metrics
    """
    mrr_scores = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
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

        # NDCG requires graded relevance scores (binary in this case)
        relevance_scores = {doc_id: 1.0 for doc_id in qa.source_ids}
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevance_scores, k))

        precision_scores.append(precision_at_k(retrieved_ids, relevant_ids, 5))
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, 10))

    sys.stdout.write(f"\r  Progress: {total_queries}/{total_queries}\n")
    sys.stdout.flush()

    # Calculate averages
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    # Latency statistics
    latency_stats = {
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "p50": statistics.median(latencies) if latencies else 0.0,
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else 0.0,
    }

    return {
        "mrr": avg_mrr,
        "ndcg@10": avg_ndcg,
        "precision@5": avg_precision,
        "recall@10": avg_recall,
        "latency_ms": latency_stats,
    }


async def build_graph_for_mode(
    mode: str,
    docs: List[SourceDoc],
) -> SmartFarmGraph:
    """Build graph for specified mode.

    Args:
        mode: Graph building mode ('rule_only', 'llm_only', 'hybrid')
        docs: Source documents

    Returns:
        Knowledge graph
    """
    print(f"\n  Building graph for mode: {mode}")
    builder = HybridGraphBuilder(mode=mode)
    graph = await builder.build(docs)
    stats = compute_graph_stats(graph)
    print(f"    Graph built: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
    return graph


def print_comparison(results: Dict[str, Dict[str, Any]]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("LLM-based Causal Extraction Graph Benchmark: Comparison")
    print("=" * 100)

    modes = list(results.keys())
    print(f"\n{'Metric':<25} " + " ".join(f"{mode:>20}" for mode in modes))
    print("-" * 100)

    # Retrieval metrics
    metrics = ["mrr", "ndcg@10", "precision@5", "recall@10"]
    for metric in metrics:
        values = [results[mode][metric] for mode in modes]
        value_strs = " ".join(f"{v:>20.4f}" for v in values)
        print(f"{metric:<25} {value_strs}")

    # Latency metrics
    print("\nLatency (ms):")
    latency_metrics = ["mean", "p50", "p95"]
    for lat_metric in latency_metrics:
        values = [results[mode]["latency_ms"][lat_metric] for mode in modes]
        value_strs = " ".join(f"{v:>20.2f}" for v in values)
        print(f"  {lat_metric:<23} {value_strs}")

    # Graph stats
    print("\nGraph Statistics:")
    graph_metrics = ["n_nodes", "n_edges"]
    for graph_metric in graph_metrics:
        values = [results[mode]["graph_stats"][graph_metric] for mode in modes]
        value_strs = " ".join(f"{v:>20}" for v in values)
        print(f"  {graph_metric:<23} {value_strs}")

    print("=" * 100)

    # Summary
    print("\nKey Findings:")
    for mode in modes:
        print(f"\n  {mode.upper()}:")
        print(f"    MRR:         {results[mode]['mrr']:.4f}")
        print(f"    NDCG@10:     {results[mode]['ndcg@10']:.4f}")
        print(f"    Precision@5: {results[mode]['precision@5']:.4f}")
        print(f"    Recall@10:   {results[mode]['recall@10']:.4f}")
        print(f"    Latency:     {results[mode]['latency_ms']['mean']:.2f}ms (mean)")
        print(f"    Graph:       {results[mode]['graph_stats']['n_nodes']} nodes, {results[mode]['graph_stats']['n_edges']} edges")


async def main():
    """Run LLM graph benchmark."""
    parser = argparse.ArgumentParser(description="LLM-based Graph Benchmark")
    parser.add_argument(
        "--benchmark-file",
        type=str,
        help="Path to benchmark JSON file (unused, using wasabi dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/llm_graph",
        help="Output directory for results",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rule_only", "llm_only", "hybrid"],
        choices=["rule_only", "llm_only", "hybrid"],
        help="Modes to test",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Top-k for retrieval evaluation",
    )
    args = parser.parse_args()

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

    print("=" * 100)
    print("LLM-based Causal Extraction Graph Benchmark")
    print("=" * 100)
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Retrieval K: {args.retrieval_k}")

    # Load data
    print("\n[1/3] Loading data...")
    corpus = load_corpus(corpus_path)
    print(f"  Loaded {len(corpus)} documents")

    qa_items = load_qa_dataset(qa_path)
    print(f"  Loaded {len(qa_items)} QA pairs")

    # Build graphs and evaluate
    print("\n[2/3] Building graphs and evaluating...")
    results = {}

    for mode in args.modes:
        try:
            # Build graph
            graph = await build_graph_for_mode(mode, corpus)
            graph_stats = compute_graph_stats(graph)

            # Create retriever
            print(f"  Creating PathRAG retriever for {mode}...")
            retriever = PathRAGLtRetriever(graph, max_hops=3, threshold=0.3, alpha=0.8)

            # Evaluate
            eval_results = evaluate_retriever(
                retriever,
                qa_items,
                k=args.retrieval_k,
                name=f"Mode: {mode}",
            )

            # Store results
            results[mode] = {
                **eval_results,
                "graph_stats": graph_stats,
            }

        except Exception as e:
            print(f"\n  ERROR in mode '{mode}': {e}")
            import traceback
            traceback.print_exc()
            # Store empty results
            results[mode] = {
                "mrr": 0.0,
                "ndcg@10": 0.0,
                "precision@5": 0.0,
                "recall@10": 0.0,
                "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
                "graph_stats": {"n_nodes": 0, "n_edges": 0, "edge_types": {}},
                "error": str(e),
            }

    # Print comparison
    print("\n[3/3] Results:")
    print_comparison(results)

    # Save results
    output_dir = era_rag_dir / "benchmarking" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "llm_graph_benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
