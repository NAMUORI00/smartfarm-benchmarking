#!/usr/bin/env python3
"""BEIR benchmark runner for dense, sparse, and RRF hybrid retrieval.

WARNING: BEIR-only benchmark for retrieval comparison (REFERENCE ONLY)
=========================================================================
This script evaluates RETRIEVAL metrics only (MRR, NDCG, Precision, Recall).
It does NOT evaluate Graph RAG capabilities:
- ❌ Entity/Relation extraction quality (RQ1)
- ❌ Multi-hop reasoning paths (RQ3)
- ❌ Answer generation quality (RQ4)

For proper Graph RAG evaluation, use:
- causal_extraction_eval.py (RQ1 - Extraction Quality)
- multihop_eval.py (RQ3 - Multi-hop Reasoning)
- ragas_eval.py (RQ4 - Answer Quality)

Edge-friendly settings with configurable retrieval methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from benchmarking.baselines import (
    DenseOnlyRetriever,
    RRFHybridRetriever,
    SparseOnlyRetriever,
    PathRAGHybridRetriever,
)
from benchmarking.metrics.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store, MiniStore


BEIR_DOWNLOAD_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


@dataclass
class BeirConfig:
    datasets: List[str]
    beir_dir: Path
    output_dir: Path
    embed_model: str
    sparse_method: str
    retrieval_k: int
    max_queries: int | None
    doc_limit: int | None
    download: bool


def parse_args() -> BeirConfig:
    parser = argparse.ArgumentParser(description="Run BEIR retrieval benchmarks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus", "arguana", "fiqa"],
        help="BEIR dataset names",
    )
    parser.add_argument(
        "--beir-dir",
        default="benchmarking/data/beir",
        help="BEIR dataset root directory",
    )
    parser.add_argument("--output-dir", default="output/beir", help="Output directory for results")
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model id",
    )
    parser.add_argument(
        "--sparse-method",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Sparse retrieval method",
    )
    parser.add_argument("--retrieval-k", type=int, default=100, help="Retrieval depth for metrics")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--doc-limit", type=int, default=None, help="Limit number of corpus documents")
    parser.add_argument("--download", action="store_true", help="Download BEIR datasets if missing")
    args = parser.parse_args()
    return BeirConfig(
        datasets=list(args.datasets),
        beir_dir=Path(args.beir_dir),
        output_dir=Path(args.output_dir),
        embed_model=args.embed_model,
        sparse_method=args.sparse_method,
        retrieval_k=args.retrieval_k,
        max_queries=args.max_queries,
        doc_limit=args.doc_limit,
        download=args.download,
    )


def ensure_dataset(dataset_name: str, beir_dir: Path, download: bool) -> Path:
    dataset_dir = beir_dir / dataset_name
    if dataset_dir.exists():
        return dataset_dir
    if not download:
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    beir_dir.mkdir(parents=True, exist_ok=True)
    dataset_url = BEIR_DOWNLOAD_URL.format(dataset=dataset_name)
    zip_path = beir_dir / f"{dataset_name}.zip"
    urllib.request.urlretrieve(dataset_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(beir_dir)
    return dataset_dir


def load_corpus(dataset_dir: Path, doc_limit: int | None = None) -> List[SourceDoc]:
    corpus_path = dataset_dir / "corpus.jsonl"
    docs: List[SourceDoc] = []
    with open(corpus_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
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
    return docs


def load_queries(dataset_dir: Path) -> Dict[str, str]:
    query_path = dataset_dir / "queries.jsonl"
    queries: Dict[str, str] = {}
    with open(query_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query_id = data.get("_id") or data.get("id")
            query_text = data.get("text") or data.get("query") or ""
            if query_id is None:
                continue
            queries[str(query_id)] = query_text
    return queries


def load_qrels(dataset_dir: Path) -> Dict[str, Dict[str, float]]:
    qrels_dir = dataset_dir / "qrels"
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
    with open(qrels_path, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle, delimiter="\t")
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
            file_handle.seek(0)
            for line in file_handle:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                query_id = parts[0]
                doc_id = parts[1]
                score = float(parts[2])
                qrels.setdefault(str(query_id), {})[str(doc_id)] = score
    return qrels


def select_queries(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    max_queries: int | None,
) -> List[Tuple[str, str]]:
    query_items = [(query_id, text) for query_id, text in queries.items() if query_id in qrels]
    if max_queries is not None:
        return query_items[:max_queries]
    return query_items


def build_retrievers(docs: List[SourceDoc], config: BeirConfig) -> Dict[str, object]:
    dense_retriever = EmbeddingRetriever(model_id=config.embed_model)
    dense_retriever.build(docs)

    if config.sparse_method == "bm25":
        sparse_store = BM25Store()
        sparse_store.index(docs)
    else:
        sparse_store = MiniStore()
        sparse_store.index(docs)

    return {
        "dense_only": DenseOnlyRetriever(dense_retriever),
        "sparse_only": SparseOnlyRetriever(sparse_store),
        "rrf_hybrid": RRFHybridRetriever(dense_retriever, sparse_store),
        "pathrag_hybrid": PathRAGHybridRetriever(dense_retriever, docs),
    }


def evaluate_method(
    method_name: str,
    retriever: object,
    query_items: List[Tuple[str, str]],
    qrels: Dict[str, Dict[str, float]],
    retrieval_k: int,
) -> Dict[str, object]:
    latency_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    mrr_values: List[float] = []
    ndcg_values: List[float] = []

    for query_id, query_text in query_items:
        start_time = time.perf_counter()
        results = retriever.search(query_text, k=retrieval_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latency_values.append(latency_ms)
        retrieved_ids = [doc.id for doc in results]
        relevance_scores = qrels.get(query_id, {})
        relevant_ids = {doc_id for doc_id, score in relevance_scores.items() if score > 0}
        precision_values.append(precision_at_k(retrieved_ids, relevant_ids, 10))
        recall_values.append(recall_at_k(retrieved_ids, relevant_ids, min(100, retrieval_k)))
        mrr_values.append(mrr(retrieved_ids, relevant_ids))
        ndcg_values.append(ndcg_at_k(retrieved_ids, relevance_scores, 10))

    return {
        "method": method_name,
        "n_queries": len(query_items),
        "latency_ms": {
            "mean": float(np.mean(latency_values)) if latency_values else 0.0,
            "p50": float(np.percentile(latency_values, 50)) if latency_values else 0.0,
            "p95": float(np.percentile(latency_values, 95)) if latency_values else 0.0,
        },
        "precision@10": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall@100": float(np.mean(recall_values)) if recall_values else 0.0,
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
        "ndcg@10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
    }


def run_dataset(dataset_name: str, config: BeirConfig) -> Dict[str, object]:
    dataset_dir = ensure_dataset(dataset_name, config.beir_dir, config.download)
    docs = load_corpus(dataset_dir, config.doc_limit)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir)
    query_items = select_queries(queries, qrels, config.max_queries)
    retrievers = build_retrievers(docs, config)

    method_results = {}
    for method_name, retriever in retrievers.items():
        method_results[method_name] = evaluate_method(
            method_name=method_name,
            retriever=retriever,
            query_items=query_items,
            qrels=qrels,
            retrieval_k=config.retrieval_k,
        )

    return {
        "dataset": dataset_name,
        "n_docs": len(docs),
        "n_queries": len(query_items),
        "embed_model": config.embed_model,
        "sparse_method": config.sparse_method,
        "retrieval_k": config.retrieval_k,
        "methods": method_results,
    }


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {}
    for dataset_name in config.datasets:
        print(f"[BEIR] Running dataset: {dataset_name}")
        dataset_result = run_dataset(dataset_name, config)
        all_results[dataset_name] = dataset_result
        output_path = config.output_dir / f"{dataset_name}_summary.json"
        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(dataset_result, file_handle, ensure_ascii=False, indent=2)

    overall_path = config.output_dir / "beir_summary.json"
    with open(overall_path, "w", encoding="utf-8") as file_handle:
        json.dump(all_results, file_handle, ensure_ascii=False, indent=2)
    print(f"[BEIR] Results saved to {overall_path}")


if __name__ == "__main__":
    main()
