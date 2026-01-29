#!/usr/bin/env python3
"""CROP PathRAG evaluation script for comparing dense, RRF hybrid, and PathRAG hybrid retrieval.

NOTE: BEIR-format retrieval comparison (REFERENCE ONLY)
=========================================================================
This script evaluates RETRIEVAL metrics (MRR, NDCG, Precision, Recall, Hit Rate)
using the CROP agricultural dataset in BEIR format.

Limitations:
- Focuses on retrieval quality, not full Graph RAG pipeline evaluation
- BEIR format lacks reference answers, limiting answer quality assessment
- For comprehensive Graph RAG evaluation, combine with:
  * causal_extraction_eval.py (entity/relation extraction quality)
  * multihop_eval.py (multi-hop reasoning paths)
  * ragas_eval.py (answer generation quality)

Supports optional RAGAS evaluation with DeepSeek-V3 for context metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from benchmarking.baselines import (
    DenseOnlyRetriever,
    RRFHybridRetriever,
    PathRAGHybridRetriever,
)
from benchmarking.metrics.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store

# Optional RAGAS support
try:
    from openai import OpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import context_precision, context_recall, answer_relevancy
    HAS_RAGAS = True
except Exception:
    HAS_RAGAS = False


@dataclass
class CropEvalConfig:
    data_dir: Path
    output_dir: Path
    embed_model: str
    top_k: int
    limit: Optional[int]
    with_ragas: bool
    deepseek_api_key: Optional[str]
    deepseek_base_url: str


def parse_args() -> CropEvalConfig:
    parser = argparse.ArgumentParser(description="CROP PathRAG evaluation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/crop",
        help="CROP dataset directory (BEIR format: corpus.jsonl, queries.jsonl, qrels/test.tsv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="output/crop_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model id",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries for testing",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Run RAGAS evaluation (requires DeepSeek-V3 API key)",
    )
    parser.add_argument(
        "--deepseek-api-key",
        default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default="https://nano-gpt.com/api/v1",
        help="DeepSeek API base URL",
    )

    args = parser.parse_args()

    api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if args.with_ragas and not api_key:
        raise ValueError("--with-ragas requires --deepseek-api-key or DEEPSEEK_API_KEY env var")

    return CropEvalConfig(
        data_dir=args.data_dir,
        output_dir=args.output,
        embed_model=args.embed_model,
        top_k=args.top_k,
        limit=args.limit,
        with_ragas=args.with_ragas,
        deepseek_api_key=api_key,
        deepseek_base_url=args.deepseek_base_url,
    )


def load_crop_corpus(data_dir: Path | str) -> List[SourceDoc]:
    """Load CROP corpus from corpus.jsonl in BEIR format."""
    data_dir = Path(data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    docs: List[SourceDoc] = []
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
            docs.append(
                SourceDoc(
                    id=str(doc_id),
                    text=merged_text,
                    metadata={"title": title}
                )
            )
    return docs


def load_crop_queries(data_dir: Path | str) -> Dict[str, str]:
    """Load CROP queries from queries.jsonl."""
    data_dir = Path(data_dir)
    query_path = data_dir / "queries.jsonl"
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    queries: Dict[str, str] = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query_id = data.get("_id") or data.get("id")
            query_text = data.get("text") or data.get("query") or ""
            if query_id:
                queries[str(query_id)] = query_text
    return queries


def load_crop_qrels(data_dir: Path | str) -> Dict[str, Dict[str, float]]:
    """Load CROP relevance judgments from qrels/test.tsv."""
    data_dir = Path(data_dir)
    qrels_path = data_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"QRels file not found: {qrels_path}")

    qrels: Dict[str, Dict[str, float]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames:
            for row in reader:
                query_id = row.get("query-id") or row.get("qid") or row.get("query_id")
                doc_id = row.get("corpus-id") or row.get("doc-id") or row.get("doc_id")
                score_value = row.get("score") or row.get("relevance") or row.get("rel")
                if query_id and doc_id and score_value:
                    score = float(score_value)
                    qrels.setdefault(str(query_id), {})[str(doc_id)] = score
        else:
            # Fallback to raw tab-separated parsing
            f.seek(0)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[1]
                    score = float(parts[2])
                    qrels.setdefault(str(query_id), {})[str(doc_id)] = score
    return qrels


def build_retrievers(docs: List[SourceDoc], embed_model: str) -> Dict[str, object]:
    """Build dense-only, RRF hybrid, and PathRAG hybrid retrievers."""
    print(f"[CROP] Building retrievers with {len(docs)} documents...")

    # Dense retriever
    dense_retriever = EmbeddingRetriever(model_id=embed_model)
    dense_retriever.build(docs)

    # Sparse retriever (BM25)
    sparse_store = BM25Store()
    sparse_store.index(docs)

    return {
        "dense_only": DenseOnlyRetriever(dense_retriever),
        "rrf_hybrid": RRFHybridRetriever(dense_retriever, sparse_store),
        "pathrag_hybrid": PathRAGHybridRetriever(dense_retriever, docs),
    }


def evaluate_method(
    retriever: object,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    top_k: int,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a single retrieval method."""
    query_items = [(qid, text) for qid, text in queries.items() if qid in qrels]
    if limit:
        query_items = query_items[:limit]

    latencies: List[float] = []
    mrr_values: List[float] = []
    ndcg_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    hit_count = 0

    for query_id, query_text in query_items:
        start_time = time.perf_counter()
        results = retriever.search(query_text, k=top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [doc.id for doc in results]
        relevance_scores = qrels.get(query_id, {})
        relevant_ids = {doc_id for doc_id, score in relevance_scores.items() if score > 0}

        # Metrics
        mrr_values.append(mrr(retrieved_ids, relevant_ids))
        ndcg_values.append(ndcg_at_k(retrieved_ids, relevance_scores, top_k))
        precision_values.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        recall_values.append(recall_at_k(retrieved_ids, relevant_ids, top_k))

        # Hit rate: at least one relevant doc in top-k
        if any(doc_id in relevant_ids for doc_id in retrieved_ids):
            hit_count += 1

    n_queries = len(query_items)
    return {
        "n_queries": n_queries,
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
        f"ndcg@{top_k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        f"precision@{top_k}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall@{top_k}": float(np.mean(recall_values)) if recall_values else 0.0,
        "hit_rate": float(hit_count / n_queries) if n_queries > 0 else 0.0,
    }


def run_ragas_evaluation(
    retrievers: Dict[str, object],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    top_k: int,
    config: CropEvalConfig,
) -> Dict[str, Any]:
    """Run RAGAS evaluation using DeepSeek-V3."""
    if not HAS_RAGAS:
        raise ImportError(
            "RAGAS dependencies not installed. "
            "Install: pip install ragas datasets langchain-community langchain-openai"
        )

    print("[CROP] Running RAGAS evaluation with DeepSeek-V3...")

    # Prepare evaluation data (using PathRAG hybrid as the default method)
    retriever = retrievers["pathrag_hybrid"]
    query_items = [(qid, text) for qid, text in queries.items() if qid in qrels]
    if config.limit:
        query_items = query_items[:config.limit]

    ragas_rows = []
    for query_id, query_text in query_items:
        results = retriever.search(query_text, k=top_k)
        contexts = [doc.text for doc in results]

        # Use first relevant document as ground truth (if available)
        relevance = qrels.get(query_id, {})
        ground_truth = ""
        for doc_id, score in relevance.items():
            if score > 0:
                ground_truth = doc_id  # Use doc_id as placeholder
                break

        ragas_rows.append({
            "question": query_text,
            "contexts": contexts,
            "ground_truth": ground_truth or "N/A",
            "answer": contexts[0] if contexts else "",  # Use top context as answer
        })

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in ragas_rows],
        "contexts": [r["contexts"] for r in ragas_rows],
        "ground_truth": [r["ground_truth"] for r in ragas_rows],
        "answer": [r["answer"] for r in ragas_rows],
    })

    # Configure DeepSeek-V3
    client = OpenAI(
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
    )
    llm = llm_factory(model="deepseek-v3-0324", client=client)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    # Run RAGAS evaluation
    metrics = [context_precision, context_recall, answer_relevancy]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )

    # Extract summary
    summary = {}
    if hasattr(result, "scores"):
        for row in result.scores:
            for key, value in row.items():
                if value is not None and isinstance(value, (int, float)):
                    if key not in summary:
                        summary[key] = []
                    summary[key].append(float(value))
        summary = {k: float(np.mean(v)) for k, v in summary.items()}

    return {
        "n_evaluated": len(ragas_rows),
        "metrics": summary,
    }


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load CROP data
    print(f"[CROP] Loading data from {config.data_dir}...")
    docs = load_crop_corpus(config.data_dir)
    queries = load_crop_queries(config.data_dir)
    qrels = load_crop_qrels(config.data_dir)

    print(f"[CROP] Loaded {len(docs)} documents, {len(queries)} queries, {len(qrels)} qrels")

    # Build retrievers
    retrievers = build_retrievers(docs, config.embed_model)

    # Evaluate each method
    method_results = {}
    for method_name, retriever in retrievers.items():
        print(f"[CROP] Evaluating {method_name}...")
        method_results[method_name] = evaluate_method(
            retriever=retriever,
            queries=queries,
            qrels=qrels,
            top_k=config.top_k,
            limit=config.limit,
        )

    # Compile results
    results = {
        "dataset": "crop",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "embed_model": config.embed_model,
        "top_k": config.top_k,
        "methods": method_results,
    }

    # Optional RAGAS evaluation
    if config.with_ragas:
        try:
            ragas_results = run_ragas_evaluation(
                retrievers=retrievers,
                queries=queries,
                qrels=qrels,
                top_k=config.top_k,
                config=config,
            )
            results["ragas"] = ragas_results
        except Exception as e:
            print(f"[CROP] RAGAS evaluation failed: {e}")
            results["ragas"] = {"error": str(e)}

    # Save results
    output_path = config.output_dir / "crop_pathrag_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[CROP] Results saved to {output_path}")
    print(f"\n[CROP] Summary:")
    print(f"  Documents: {len(docs)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Top-K: {config.top_k}")
    print(f"\n  Method Performance:")
    for method_name, metrics in method_results.items():
        print(f"    {method_name}:")
        print(f"      MRR: {metrics['mrr']:.4f}")
        print(f"      NDCG@{config.top_k}: {metrics[f'ndcg@{config.top_k}']:.4f}")
        print(f"      Precision@{config.top_k}: {metrics[f'precision@{config.top_k}']:.4f}")
        print(f"      Recall@{config.top_k}: {metrics[f'recall@{config.top_k}']:.4f}")
        print(f"      Hit Rate: {metrics['hit_rate']:.4f}")

    if "ragas" in results and "metrics" in results["ragas"]:
        print(f"\n  RAGAS Metrics (DeepSeek-V3):")
        for metric_name, value in results["ragas"]["metrics"].items():
            print(f"    {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
