#!/usr/bin/env python3
"""LightRAG 그래프 구축 및 테스트

문서를 LightRAG에 삽입하여 그래프를 구축하고,
그래프 기반 검색 성능을 Sparse-only와 비교합니다.

사용 예시:
    python -m benchmarking.experiments.lightrag_graph_test \
        --doc-limit 100 \
        --qa-limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

os.environ.setdefault('LLMLITE_HOST', 'http://localhost:45857')

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Config.Settings import settings
from core.Services.Retrieval.Sparse import BM25Store, MiniStore
from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.LightRAG import LightRAGRetriever

from benchmarking.baselines import SparseOnlyRetriever
from benchmarking.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CORPUS_PATH = _WORKSPACE_ROOT / "smartfarm-ingest" / "output" / "wasabi_en_ko_parallel.jsonl"
_DEFAULT_QA_PATH = _WORKSPACE_ROOT / "smartfarm-ingest" / "output" / "wasabi_qa_dataset.jsonl"


@dataclass
class QAItem:
    id: str
    question: str
    answer: str
    source_ids: List[str] = field(default_factory=list)


def load_qa_dataset(path: Path, limit: int = None) -> List[QAItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(QAItem(
                id=data["id"],
                question=data["question"],
                answer=data["answer"],
                source_ids=data.get("source_ids", []),
            ))
            if limit and len(items) >= limit:
                break
    return items


def load_corpus(path: Path, limit: int = None) -> List[SourceDoc]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            docs.append(SourceDoc(
                id=data["id"],
                text=data.get("text_ko") or data.get("text", ""),
                metadata=data.get("metadata", {}),
            ))
            if limit and len(docs) >= limit:
                break
    return docs


def evaluate_retriever(retriever, qa_items: List[QAItem], top_k: int = 4, name: str = "") -> Dict[str, Any]:
    """리트리버 평가"""
    latencies = []
    p_at_k_list = []
    r_at_k_list = []
    mrr_list = []
    ndcg_list = []

    for i, qa in enumerate(qa_items):
        start = time.perf_counter()
        results = retriever.search(qa.question, k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [r.id for r in results]
        relevant_ids = set(qa.source_ids) if qa.source_ids else set()

        if relevant_ids:
            p_at_k_list.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
            r_at_k_list.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
            mrr_list.append(mrr(retrieved_ids, relevant_ids))
            relevance_scores = {rid: 1.0 for rid in relevant_ids}
            ndcg_list.append(ndcg_at_k(retrieved_ids, relevance_scores, top_k))
        else:
            p_at_k_list.append(0.0)
            r_at_k_list.append(0.0)
            mrr_list.append(0.0)
            ndcg_list.append(0.0)

        if (i + 1) % 20 == 0:
            print(f"  [{name}] Processed {i + 1}/{len(qa_items)} queries")

    return {
        "n_queries": len(qa_items),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
        },
        "precision@k": {"mean": float(np.mean(p_at_k_list)), "std": float(np.std(p_at_k_list))},
        "recall@k": {"mean": float(np.mean(r_at_k_list)), "std": float(np.std(r_at_k_list))},
        "mrr": {"mean": float(np.mean(mrr_list)), "std": float(np.std(mrr_list))},
        "ndcg@k": {"mean": float(np.mean(ndcg_list)), "std": float(np.std(ndcg_list))},
    }


def main():
    parser = argparse.ArgumentParser(description="LightRAG Graph Test")
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS_PATH)
    parser.add_argument("--qa-file", type=Path, default=_DEFAULT_QA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/experiments/lightrag_graph"))
    parser.add_argument("--doc-limit", type=int, default=100, help="Number of documents to index")
    parser.add_argument("--qa-limit", type=int, default=50, help="Number of QA items to test")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lightrag_dir = args.output_dir / "lightrag_workdir"

    # Clean previous workdir
    if lightrag_dir.exists():
        shutil.rmtree(lightrag_dir)

    # Load data
    print(f"[1/5] Loading corpus (limit={args.doc_limit})...")
    corpus = load_corpus(args.corpus, limit=args.doc_limit)
    print(f"  Loaded {len(corpus)} documents")

    print(f"[2/5] Loading QA dataset (limit={args.qa_limit})...")
    qa_items = load_qa_dataset(args.qa_file, limit=args.qa_limit)
    print(f"  Loaded {len(qa_items)} QA items")

    # Build Sparse index for comparison
    print("[3/5] Building Sparse index...")
    sparse_method = getattr(settings, "SPARSE_METHOD", "bm25").lower()
    sparse = MiniStore() if sparse_method == "tfidf" else BM25Store()
    sparse.index(corpus)
    sparse_retriever = SparseOnlyRetriever(sparse)

    # Build LightRAG graph
    print(f"[4/5] Building LightRAG graph ({len(corpus)} documents)...")
    print("  This will take time due to LLM-based entity extraction...")

    lightrag = LightRAGRetriever(
        working_dir=lightrag_dir,
        query_mode="hybrid",
    )
    lightrag.initialize()

    # Use batch insert (single event loop session)
    start_time = time.time()
    print(f"  Inserting {len(corpus)} documents in batch mode...")

    # Use insert_docs which internally uses batch insert
    lightrag.insert_docs(corpus)

    build_time = time.time() - start_time
    print(f"  Graph built in {build_time/60:.1f} minutes")

    # Evaluate
    print("\n[5/5] Evaluating retrievers...")
    results = {}

    print("\n=== Sparse-only ===")
    results["sparse_only"] = evaluate_retriever(sparse_retriever, qa_items, args.top_k, "Sparse")
    print(f"  MRR: {results['sparse_only']['mrr']['mean']:.4f}")
    print(f"  Recall@{args.top_k}: {results['sparse_only']['recall@k']['mean']:.4f}")

    print("\n=== LightRAG (Graph + Keyword) ===")
    results["lightrag_graph"] = evaluate_retriever(lightrag, qa_items, args.top_k, "LightRAG")
    print(f"  MRR: {results['lightrag_graph']['mrr']['mean']:.4f}")
    print(f"  Recall@{args.top_k}: {results['lightrag_graph']['recall@k']['mean']:.4f}")

    # Add build info
    results["build_info"] = {
        "doc_count": len(corpus),
        "build_time_sec": build_time,
    }

    # Save results
    result_path = args.output_dir / "lightrag_graph_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {result_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: LightRAG Graph vs Sparse-only")
    print("=" * 60)
    print(f"Documents indexed: {len(corpus)}")
    print(f"QA items tested: {len(qa_items)}")
    print(f"Graph build time: {build_time/60:.1f} minutes")
    print("-" * 60)
    print(f"{'Method':<20} {'MRR':>10} {'Recall@4':>10} {'Latency':>12}")
    print("-" * 60)
    for method in ["sparse_only", "lightrag_graph"]:
        data = results[method]
        print(f"{method:<20} {data['mrr']['mean']:>10.4f} {data['recall@k']['mean']:>10.4f} {data['latency_ms']['mean']:>10.2f}ms")
    print("=" * 60)

    # Winner
    if results["lightrag_graph"]["mrr"]["mean"] > results["sparse_only"]["mrr"]["mean"]:
        improvement = ((results["lightrag_graph"]["mrr"]["mean"] / results["sparse_only"]["mrr"]["mean"]) - 1) * 100
        print(f"\n[WINNER] LightRAG wins! (+{improvement:.1f}% MRR)")
    else:
        if results["lightrag_graph"]["mrr"]["mean"] > 0:
            decline = ((results["sparse_only"]["mrr"]["mean"] / results["lightrag_graph"]["mrr"]["mean"]) - 1) * 100
            print(f"\n[RESULT] Sparse-only still better ({decline:.1f}% higher MRR)")
        else:
            print(f"\n[RESULT] Sparse-only wins (LightRAG MRR=0)")


if __name__ == "__main__":
    main()
