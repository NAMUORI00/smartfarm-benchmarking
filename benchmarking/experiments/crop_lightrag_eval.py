#!/usr/bin/env python3
"""LightRAG comparison evaluation on CROP dataset.

Evaluates LightRAG (EMNLP 2025) baseline against CROP agricultural QA benchmark
using same metrics as PathRAG evaluation for fair comparison.

NOTE: BEIR-format retrieval comparison (REFERENCE ONLY)
=========================================================================
This script evaluates RETRIEVAL metrics (MRR, NDCG, Precision, Recall, Hit Rate)
for LightRAG baseline using the CROP dataset in BEIR format.

Limitations:
- Focuses on retrieval quality, not full Graph RAG pipeline evaluation
- BEIR format lacks reference answers, limiting answer quality assessment
- For comprehensive Graph RAG evaluation, combine with:
  * causal_extraction_eval.py (entity/relation extraction quality)
  * multihop_eval.py (multi-hop reasoning paths)
  * ragas_eval.py (answer generation quality)

Usage:
    python -m benchmarking.experiments.crop_lightrag_eval \
        --data-dir data/crop \
        --output output/crop_lightrag_eval \
        --limit 1000 \
        --top-k 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from benchmarking.baselines.lightrag import LightRAGBaseline
from benchmarking.metrics.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    hit_rate,
)
from core.Models.Schemas import SourceDoc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_crop_corpus(data_dir: Path, doc_limit: int | None = None) -> List[SourceDoc]:
    """Load CROP corpus from BEIR-format JSONL.

    Args:
        data_dir: Path to CROP data directory
        doc_limit: Optional limit on number of documents

    Returns:
        List of SourceDoc objects
    """
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    docs: List[SourceDoc] = []
    logger.info(f"Loading corpus from {corpus_path}...")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get("_id") or data.get("id")
            title = data.get("title", "")
            text = data.get("text", "")
            merged_text = "\n".join([part for part in [title, text] if part])
            docs.append(SourceDoc(id=str(doc_id), text=merged_text, metadata={"title": title}))

            if doc_limit is not None and len(docs) >= doc_limit:
                break

    logger.info(f"Loaded {len(docs)} documents from corpus")
    return docs


def load_crop_queries(data_dir: Path) -> Dict[str, str]:
    """Load CROP queries from BEIR-format JSONL.

    Args:
        data_dir: Path to CROP data directory

    Returns:
        Dict mapping query_id -> query_text
    """
    queries_path = data_dir / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    queries: Dict[str, str] = {}
    logger.info(f"Loading queries from {queries_path}...")

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query_id = data.get("_id") or data.get("id")
            query_text = data.get("text") or data.get("query") or ""
            if query_id is None:
                continue
            queries[str(query_id)] = query_text

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_crop_qrels(data_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load CROP qrels (query-document relevance judgments).

    Args:
        data_dir: Path to CROP data directory

    Returns:
        Dict mapping query_id -> {doc_id -> relevance_score}
    """
    qrels_dir = data_dir / "qrels"
    if not qrels_dir.exists():
        raise FileNotFoundError(f"Qrels directory not found: {qrels_dir}")

    candidate_files = ["test.tsv", "dev.tsv", "train.tsv"]
    qrels_path = None
    for filename in candidate_files:
        path_candidate = qrels_dir / filename
        if path_candidate.exists():
            qrels_path = path_candidate
            break

    if qrels_path is None:
        raise FileNotFoundError(f"No qrels file found in {qrels_dir}")

    qrels: Dict[str, Dict[str, float]] = {}
    logger.info(f"Loading qrels from {qrels_path}...")

    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames:
            for row in reader:
                query_id = row.get("query-id") or row.get("qid") or row.get("query_id")
                doc_id = row.get("corpus-id") or row.get("doc-id") or row.get("doc_id")
                score_value = row.get("score") or row.get("relevance") or row.get("rel")
                if query_id is None or doc_id is None or score_value is None:
                    continue
                score = float(score_value)
                qrels.setdefault(str(query_id), {})[str(doc_id)] = score
        else:
            f.seek(0)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                query_id = parts[0]
                doc_id = parts[1]
                score = float(parts[2])
                qrels.setdefault(str(query_id), {})[str(doc_id)] = score

    logger.info(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def build_lightrag(
    docs: List[SourceDoc],
    working_dir: Path,
) -> LightRAGBaseline:
    """Build LightRAG graph from CROP corpus.

    Args:
        docs: List of source documents
        working_dir: Working directory for LightRAG storage

    Returns:
        LightRAGBaseline retriever instance
    """
    logger.info(f"Building LightRAG graph with {len(docs)} documents...")
    logger.info(f"Working directory: {working_dir}")

    start_time = time.perf_counter()

    # Build LightRAG with hybrid mode (same as PathRAG for fair comparison)
    lightrag = LightRAGBaseline.build_from_docs(
        docs=docs,
        working_dir=working_dir,
        dense_retriever=None,  # No dense fallback
        query_mode="hybrid",  # Use hybrid mode (local + global)
    )

    build_time = time.perf_counter() - start_time
    logger.info(f"LightRAG graph built in {build_time:.2f} seconds")

    # Get graph statistics if available
    try:
        graph_stats = get_graph_stats(working_dir)
        logger.info(f"Graph stats: {graph_stats}")
    except Exception as e:
        logger.warning(f"Failed to get graph stats: {e}")

    return lightrag


def get_graph_stats(working_dir: Path) -> Dict[str, int]:
    """Extract graph statistics from LightRAG working directory.

    Args:
        working_dir: LightRAG working directory

    Returns:
        Dict with graph statistics (n_nodes, n_edges, etc.)
    """
    stats = {}

    # Try to load graph structure files
    graph_chunk_entity_path = working_dir / "graph_chunk_entity_relation.graphml"
    if graph_chunk_entity_path.exists():
        # Parse GraphML to count nodes and edges
        try:
            with open(graph_chunk_entity_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple counting (not parsing full XML for efficiency)
                stats["n_nodes"] = content.count("<node id=")
                stats["n_edges"] = content.count("<edge source=")
        except Exception as e:
            logger.debug(f"Failed to parse GraphML: {e}")

    # Count other storage files
    kv_store_path = working_dir / "kv_store_full_docs.json"
    if kv_store_path.exists():
        try:
            with open(kv_store_path, "r", encoding="utf-8") as f:
                kv_data = json.load(f)
                stats["n_docs"] = len(kv_data)
        except Exception as e:
            logger.debug(f"Failed to load kv_store: {e}")

    return stats


def evaluate_lightrag(
    lightrag: LightRAGBaseline,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    top_k: int = 4,
    max_queries: int | None = None,
) -> Dict[str, object]:
    """Evaluate LightRAG on CROP dataset.

    Args:
        lightrag: LightRAG retriever instance
        queries: Dict of query_id -> query_text
        qrels: Dict of query_id -> {doc_id -> relevance_score}
        top_k: Number of documents to retrieve (default: 4, per paper)
        max_queries: Optional limit on number of queries to evaluate

    Returns:
        Dict with evaluation metrics
    """
    logger.info(f"Evaluating LightRAG with top_k={top_k}...")

    # Select queries that have relevance judgments
    query_items = [(qid, text) for qid, text in queries.items() if qid in qrels]

    if max_queries is not None:
        query_items = query_items[:max_queries]

    logger.info(f"Evaluating on {len(query_items)} queries")

    # Per-query metric lists
    mrr_values: List[float] = []
    ndcg_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    hit_rate_values: List[float] = []
    latency_values: List[float] = []

    for i, (query_id, query_text) in enumerate(query_items):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(query_items)} queries evaluated")

        # Retrieve documents
        start_time = time.perf_counter()
        try:
            results = lightrag.search(query_text, k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latency_values.append(latency_ms)
        except Exception as e:
            logger.warning(f"Search failed for query {query_id}: {e}")
            # Use empty results for failed queries
            results = []
            latency_values.append(0.0)

        # Get retrieved document IDs
        retrieved_ids = [doc.id for doc in results]

        # Get relevance judgments for this query
        relevance_scores = qrels.get(query_id, {})
        relevant_ids = {doc_id for doc_id, score in relevance_scores.items() if score > 0}

        # Compute metrics
        mrr_values.append(mrr(retrieved_ids, relevant_ids))
        ndcg_values.append(ndcg_at_k(retrieved_ids, relevance_scores, top_k))
        precision_values.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        recall_values.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        hit_rate_values.append(hit_rate(retrieved_ids, relevant_ids, top_k))

    # Aggregate metrics
    metrics = {
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
        f"ndcg@{top_k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        f"precision@{top_k}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall@{top_k}": float(np.mean(recall_values)) if recall_values else 0.0,
        f"hit_rate@{top_k}": float(np.mean(hit_rate_values)) if hit_rate_values else 0.0,
    }

    # Also compute @10 for compatibility with BEIR benchmarks
    if top_k != 10:
        ndcg_10_values: List[float] = []
        for query_id, _ in query_items:
            relevance_scores = qrels.get(query_id, {})
            # Use already retrieved results (may be < 10)
            try:
                results = lightrag.search(query_text, k=10)
                retrieved_ids = [doc.id for doc in results]
                ndcg_10_values.append(ndcg_at_k(retrieved_ids, relevance_scores, 10))
            except Exception:
                ndcg_10_values.append(0.0)

        metrics["ndcg@10"] = float(np.mean(ndcg_10_values)) if ndcg_10_values else 0.0

    # Latency statistics
    latency_stats = {
        "mean_ms": float(np.mean(latency_values)) if latency_values else 0.0,
        "p50_ms": float(np.percentile(latency_values, 50)) if latency_values else 0.0,
        "p95_ms": float(np.percentile(latency_values, 95)) if latency_values else 0.0,
    }

    logger.info(f"Evaluation complete. Metrics: {metrics}")

    return {
        "metrics": metrics,
        "latency": latency_stats,
        "n_queries_evaluated": len(query_items),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LightRAG on CROP agricultural QA dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/crop",
        help="Path to CROP data directory (default: data/crop)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/crop_lightrag_eval",
        help="Output directory for results (default: output/crop_lightrag_eval)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of corpus documents (for testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4, per paper)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (for testing)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="LightRAG working directory (default: <output>/lightrag_workdir)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of LightRAG graph (ignore existing)",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set working directory
    if args.working_dir:
        working_dir = Path(args.working_dir)
    else:
        working_dir = output_dir / "lightrag_workdir"

    logger.info("="*60)
    logger.info("CROP Dataset - LightRAG Evaluation")
    logger.info("="*60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Top-k: {args.top_k}")

    # Load CROP dataset
    docs = load_crop_corpus(data_dir, doc_limit=args.limit)
    queries = load_crop_queries(data_dir)
    qrels = load_crop_qrels(data_dir)

    # Build or load LightRAG
    if args.rebuild or not working_dir.exists():
        lightrag = build_lightrag(docs, working_dir)
    else:
        logger.info(f"Loading existing LightRAG from {working_dir}")
        try:
            lightrag = LightRAGBaseline.from_working_dir(
                working_dir=working_dir,
                docs=docs,
                query_mode="hybrid",
            )
            logger.info("LightRAG loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load existing LightRAG: {e}")
            logger.info("Rebuilding LightRAG...")
            lightrag = build_lightrag(docs, working_dir)

    # Evaluate
    eval_results = evaluate_lightrag(
        lightrag=lightrag,
        queries=queries,
        qrels=qrels,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )

    # Get graph statistics
    graph_stats = get_graph_stats(working_dir)

    # Prepare output
    output_data = {
        "dataset": "crop",
        "method": "lightrag",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_queries_evaluated": eval_results["n_queries_evaluated"],
        "top_k": args.top_k,
        "metrics": eval_results["metrics"],
        "latency": eval_results["latency"],
        "graph_stats": graph_stats,
        "config": {
            "query_mode": "hybrid",
            "data_dir": str(data_dir),
            "working_dir": str(working_dir),
        },
    }

    # Save results
    output_path = output_dir / "crop_lightrag_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info("="*60)
    logger.info(f"Results saved to {output_path}")
    logger.info("="*60)
    logger.info("\nMetrics Summary:")
    for metric_name, metric_value in eval_results["metrics"].items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info(f"\nLatency (mean): {eval_results['latency']['mean_ms']:.2f} ms")


if __name__ == "__main__":
    main()
