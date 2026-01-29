#!/usr/bin/env python3
"""Multi-hop reasoning evaluation for RQ3.

Evaluates multi-hop reasoning capabilities of knowledge graph-based retrieval:
- Loads gold multi-hop annotations (question -> reasoning path -> answer)
- Builds knowledge graph using HybridGraphBuilder
- Performs multi-hop retrieval using PathRAGLtRetriever
- Computes hop-level accuracy, path exact match, supporting facts evaluation
- Stratifies results by question type and number of hops
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarking.metrics.multihop_metrics import (
    GoldMultihopAnnotation,
    HopResult,
    MultihopMetrics,
    MultihopPath,
)
from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models import SourceDoc
from core.Models.Graph import SmartFarmGraph
from core.Services.Ingest.HybridGraphBuilder import HybridGraphBuilder
from core.Services.Retrieval.PathRAG import PathRAGLtRetriever


def load_gold_annotations(path: Path, limit: Optional[int] = None) -> List[GoldMultihopAnnotation]:
    """Load gold multi-hop annotations from JSONL file.

    Expected format per line:
    {
        "question_id": "q1",
        "question": "What causes powdery mildew in tomatoes?",
        "gold_hops": [
            {
                "hop_index": 0,
                "source_node": "env:high_humidity",
                "edge_type": "causes",
                "target_node": "disease:powdery_mildew",
                "confidence": 1.0
            },
            {
                "hop_index": 1,
                "source_node": "disease:powdery_mildew",
                "edge_type": "affects",
                "target_node": "crop:tomato",
                "confidence": 1.0
            }
        ],
        "gold_answer": "High humidity causes powdery mildew in tomatoes",
        "gold_supporting_facts": ["env:high_humidity", "disease:powdery_mildew", "crop:tomato"],
        "question_type": "causal_chain",
        "num_hops": 2
    }

    Args:
        path: Path to gold annotations JSONL file
        limit: Maximum number of annotations to load

    Returns:
        List of GoldMultihopAnnotation objects
    """
    annotations: List[GoldMultihopAnnotation] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Parse hop results
            hops = [
                HopResult(
                    hop_index=h["hop_index"],
                    source_node=h["source_node"],
                    edge_type=h["edge_type"],
                    target_node=h["target_node"],
                    confidence=h.get("confidence", 1.0),
                )
                for h in data.get("gold_hops", [])
            ]

            annotation = GoldMultihopAnnotation(
                question_id=data["question_id"],
                question=data["question"],
                gold_hops=hops,
                gold_answer=data.get("gold_answer", ""),
                gold_supporting_facts=data.get("gold_supporting_facts", []),
                question_type=data.get("question_type"),
                num_hops=data.get("num_hops"),
            )
            annotations.append(annotation)

            if limit and len(annotations) >= limit:
                break

    return annotations


def load_corpus(path: Path) -> List[SourceDoc]:
    """Load document corpus from JSONL file.

    Expected format per line:
    {
        "id": "doc1",
        "text": "Document text...",
        "metadata": {...}
    }

    Args:
        path: Path to corpus JSONL file

    Returns:
        List of SourceDoc objects
    """
    docs: List[SourceDoc] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            doc = SourceDoc(
                id=data["id"],
                text=data["text"],
                metadata=data.get("metadata", {}),
            )
            docs.append(doc)

    return docs


async def build_knowledge_graph(
    docs: List[SourceDoc],
    graph_mode: str,
) -> SmartFarmGraph:
    """Build knowledge graph using HybridGraphBuilder.

    Args:
        docs: Source documents
        graph_mode: Graph building mode (rule_only, llm_only, hybrid)

    Returns:
        Knowledge graph
    """
    builder = HybridGraphBuilder(mode=graph_mode)
    graph = await builder.build(docs)

    print(f"Knowledge graph built ({graph_mode} mode):")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")

    return graph


def extract_multihop_path_from_retrieval(
    question: str,
    retrieved_docs: List[SourceDoc],
    graph: SmartFarmGraph,
    max_hops: int,
) -> MultihopPath:
    """Extract multi-hop reasoning path from retrieved documents.

    This is a simplified extraction - in production, you'd need more sophisticated
    path reconstruction from the retriever's internal state.

    For now, we reconstruct hops by analyzing the retrieved documents' metadata
    and graph structure.

    Args:
        question: Input question
        retrieved_docs: Documents retrieved by PathRAGLtRetriever
        graph: Knowledge graph
        max_hops: Maximum number of hops

    Returns:
        MultihopPath with reconstructed reasoning path
    """
    # Extract supporting facts (node IDs) from retrieved documents
    supporting_facts = [doc.id for doc in retrieved_docs]

    # Reconstruct hops by tracing edges between supporting facts
    # This is a simplification - PathRAG could expose its internal path state
    hops: List[HopResult] = []

    # For each pair of consecutive supporting facts, find connecting edge
    for i in range(min(len(supporting_facts) - 1, max_hops)):
        source_id = supporting_facts[i]
        target_id = supporting_facts[i + 1]

        # Find edge connecting these nodes
        connecting_edge = None
        for edge in graph.edges:
            if (edge.source == source_id and edge.target == target_id) or \
               (edge.target == source_id and edge.source == target_id):
                connecting_edge = edge
                break

        if connecting_edge:
            hops.append(HopResult(
                hop_index=i,
                source_node=source_id,
                edge_type=connecting_edge.type,
                target_node=target_id,
                confidence=connecting_edge.weight,
            ))

    # Extract final answer from first retrieved document
    final_answer = retrieved_docs[0].text if retrieved_docs else ""

    return MultihopPath(
        question=question,
        hops=hops,
        final_answer=final_answer,
        supporting_facts=supporting_facts,
    )


async def run_multihop_evaluation(
    gold_annotations: List[GoldMultihopAnnotation],
    graph: SmartFarmGraph,
    retrieval_mode: str,
    max_hops: int,
    top_k: int,
    stratify_by_type: bool,
    stratify_by_hops: bool,
) -> Dict[str, Any]:
    """Run multi-hop evaluation on all annotations.

    Args:
        gold_annotations: Gold standard annotations
        graph: Knowledge graph
        retrieval_mode: Retrieval mode (dense_only, pathrag_only, hybrid)
        max_hops: Maximum number of hops
        top_k: Number of documents to retrieve
        stratify_by_type: Whether to stratify results by question type
        stratify_by_hops: Whether to stratify results by number of hops

    Returns:
        Evaluation results dictionary
    """
    # Initialize retriever based on mode
    if retrieval_mode == "pathrag_only":
        retriever = PathRAGLtRetriever(
            graph=graph,
            max_hops=max_hops,
            threshold=0.1,
            alpha=0.8,
            max_results=top_k,
            seed_match_mode="keyword",
        )
    elif retrieval_mode == "dense_only":
        # For dense_only, we'd need a vector retriever
        # Placeholder: use PathRAG with max coverage
        retriever = PathRAGLtRetriever(
            graph=graph,
            max_hops=1,  # Shallow for dense-like behavior
            threshold=0.0,
            alpha=1.0,
            max_results=top_k,
            seed_match_mode="all",
        )
    else:  # hybrid
        # Hybrid would combine both - for now use PathRAG
        retriever = PathRAGLtRetriever(
            graph=graph,
            max_hops=max_hops,
            threshold=0.1,
            alpha=0.8,
            max_results=top_k,
            seed_match_mode="keyword",
        )

    # Run retrieval for each question
    predictions: List[MultihopPath] = []

    for gold_ann in gold_annotations:
        question = gold_ann.question

        # Retrieve documents
        retrieved_docs = retriever.search(question, k=top_k)

        # Extract reasoning path from retrieval results
        pred_path = extract_multihop_path_from_retrieval(
            question=question,
            retrieved_docs=retrieved_docs,
            graph=graph,
            max_hops=max_hops,
        )

        predictions.append(pred_path)

    # Compute metrics
    metrics_computer = MultihopMetrics(strict_hop_matching=True)

    # Overall metrics
    all_metrics = [
        metrics_computer.compute_all(pred, gold)
        for pred, gold in zip(predictions, gold_annotations)
    ]
    overall_metrics = metrics_computer.aggregate(all_metrics)

    results = {
        "num_questions": len(gold_annotations),
        "retrieval_mode": retrieval_mode,
        "max_hops": max_hops,
        "top_k": top_k,
        "overall_metrics": overall_metrics,
    }

    # Stratify by question type if requested
    if stratify_by_type:
        by_type = metrics_computer.compute_by_question_type(predictions, gold_annotations)
        results["by_question_type"] = by_type

    # Stratify by number of hops if requested
    if stratify_by_hops:
        by_hops = metrics_computer.compute_by_num_hops(predictions, gold_annotations)
        # Convert int keys to strings for JSON serialization
        results["by_num_hops"] = {str(k): v for k, v in by_hops.items()}

    # Include per-question results (first 10 for brevity)
    results["sample_predictions"] = [
        {
            "question_id": gold.question_id,
            "question": gold.question,
            "predicted_hops": [asdict(h) for h in pred.hops],
            "gold_hops": [asdict(h) for h in gold.gold_hops],
            "metrics": metrics_computer.compute_all(pred, gold),
        }
        for pred, gold in zip(predictions[:10], gold_annotations[:10])
    ]

    return results


async def main_async() -> None:
    """Async main function."""
    ap = argparse.ArgumentParser(description="Multi-hop reasoning evaluation (RQ3)")
    ap.add_argument("--gold-file", type=Path, required=True, help="Gold annotations JSONL path")
    ap.add_argument("--corpus-file", type=Path, required=True, help="Document corpus JSONL path")
    ap.add_argument("--output", type=Path, required=True, help="Results JSON output path")
    ap.add_argument("--max-hops", type=int, default=3, help="Maximum number of hops (default: 3)")
    ap.add_argument(
        "--graph-mode",
        choices=["rule_only", "llm_only", "hybrid"],
        default="hybrid",
        help="Graph building mode (default: hybrid)",
    )
    ap.add_argument(
        "--retrieval-mode",
        choices=["dense_only", "pathrag_only", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    ap.add_argument("--top-k", type=int, default=4, help="Number of documents to retrieve (default: 4)")
    ap.add_argument(
        "--stratify-by-type",
        action="store_true",
        help="Compute metrics by question type",
    )
    ap.add_argument(
        "--stratify-by-hops",
        action="store_true",
        help="Compute metrics by number of hops",
    )
    args = ap.parse_args()

    # Load gold annotations
    print(f"Loading gold annotations from {args.gold_file}...")
    gold_annotations = load_gold_annotations(args.gold_file, limit=args.limit)
    print(f"Loaded {len(gold_annotations)} annotations")

    # Load corpus
    print(f"Loading corpus from {args.corpus_file}...")
    docs = load_corpus(args.corpus_file)
    print(f"Loaded {len(docs)} documents")

    # Build knowledge graph
    print(f"Building knowledge graph ({args.graph_mode} mode)...")
    graph = await build_knowledge_graph(docs, graph_mode=args.graph_mode)

    # Run evaluation
    print(f"Running multi-hop evaluation ({args.retrieval_mode} retrieval)...")
    results = await run_multihop_evaluation(
        gold_annotations=gold_annotations,
        graph=graph,
        retrieval_mode=args.retrieval_mode,
        max_hops=args.max_hops,
        top_k=args.top_k,
        stratify_by_type=args.stratify_by_type,
        stratify_by_hops=args.stratify_by_hops,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n=== Multi-hop Evaluation Summary ===")
    print(f"Questions evaluated: {results['num_questions']}")
    print(f"Retrieval mode: {results['retrieval_mode']}")
    print(f"Max hops: {results['max_hops']}")
    print("\nOverall Metrics:")
    for metric_name, value in sorted(results["overall_metrics"].items()):
        print(f"  {metric_name}: {value:.4f}")

    if args.stratify_by_type and "by_question_type" in results:
        print("\nBy Question Type:")
        for qtype, metrics in sorted(results["by_question_type"].items()):
            print(f"\n  {qtype}:")
            for metric_name, value in sorted(metrics.items()):
                print(f"    {metric_name}: {value:.4f}")

    if args.stratify_by_hops and "by_num_hops" in results:
        print("\nBy Number of Hops:")
        for num_hops, metrics in sorted(results["by_num_hops"].items(), key=lambda x: int(x[0])):
            print(f"\n  {num_hops} hops:")
            for metric_name, value in sorted(metrics.items()):
                print(f"    {metric_name}: {value:.4f}")


def main() -> None:
    """Synchronous entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
