#!/usr/bin/env python3
"""LightRAG vs Sparse-only 비교 실험

LightRAG와 기존 베이스라인의 검색 성능 비교.

LightRAG 특성:
- 문서 삽입 시 LLM으로 엔티티/관계 추출 (시간 소요)
- 그래프 구축 후 다양한 검색 모드 지원
- LightRAGRetrieverWithDense: Dense + 키워드 결합 버전

실험 전략:
1. 소규모 샘플로 빠른 비교
2. LightRAG 키워드 검색 vs Sparse-only 비교
3. (선택) Dense + LightRAG 통합 검색

사용 예시:
    python -m benchmarking.experiments.lightrag_comparison \
        --sample-size 50 \
        --output-dir output/experiments/lightrag
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store, MiniStore
from core.Config.Settings import settings
from core.Models.Schemas import SourceDoc

# 베이스라인
from benchmarking.baselines import (
    SparseOnlyRetriever,
    DenseOnlyRetriever,
    RRFHybridRetriever,
    LightRAGBaseline,
)

# 메트릭
from benchmarking.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
)


@dataclass
class QAItem:
    """QA 데이터셋 항목"""
    id: str
    question: str
    answer: str
    source_ids: List[str] = field(default_factory=list)


def load_qa_dataset(path: Path, limit: int = None) -> List[QAItem]:
    """QA 데이터셋 로드"""
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


def load_corpus(path: Path) -> List[SourceDoc]:
    """말뭉치 로드"""
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
    return docs


def evaluate_retriever(retriever, qa_items: List[QAItem], top_k: int = 4) -> Dict[str, Any]:
    """리트리버 평가"""
    latencies = []
    p_at_k_list = []
    r_at_k_list = []
    mrr_list = []
    ndcg_list = []

    for qa in qa_items:
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

    return {
        "n_queries": len(qa_items),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "precision@k": {"mean": float(np.mean(p_at_k_list)), "std": float(np.std(p_at_k_list))},
        "recall@k": {"mean": float(np.mean(r_at_k_list)), "std": float(np.std(r_at_k_list))},
        "mrr": {"mean": float(np.mean(mrr_list)), "std": float(np.std(mrr_list))},
        "ndcg@k": {"mean": float(np.mean(ndcg_list)), "std": float(np.std(ndcg_list))},
    }


def run_comparison(
    corpus_path: Path,
    qa_path: Path,
    output_dir: Path,
    sample_size: int = 50,
    top_k: int = 4,
):
    """LightRAG vs 베이스라인 비교"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print(f"[LightRAG Comparison] Loading corpus from {corpus_path}")
    corpus = load_corpus(corpus_path)
    print(f"  Loaded {len(corpus)} documents")

    print(f"[LightRAG Comparison] Loading QA dataset from {qa_path} (sample={sample_size})")
    qa_items = load_qa_dataset(qa_path, limit=sample_size)
    print(f"  Loaded {len(qa_items)} QA items")

    results = {}

    # 1. Dense index 구축
    print("\n[1/4] Building Dense index...")
    dense = EmbeddingRetriever()
    dense.build(corpus)

    # 2. Sparse index 구축
    print("[2/4] Building Sparse index...")
    sparse_method = getattr(settings, "SPARSE_METHOD", "bm25").lower()
    sparse = MiniStore() if sparse_method == "tfidf" else BM25Store()
    sparse.index(corpus)

    # 3. 베이스라인 평가
    print("\n=== Evaluating Baselines ===")

    # Sparse-only
    print("\n[Sparse-only]")
    sparse_retriever = SparseOnlyRetriever(sparse)
    results["sparse_only"] = evaluate_retriever(sparse_retriever, qa_items, top_k)
    print(f"  MRR: {results['sparse_only']['mrr']['mean']:.4f}")
    print(f"  Recall@{top_k}: {results['sparse_only']['recall@k']['mean']:.4f}")
    print(f"  Latency: {results['sparse_only']['latency_ms']['mean']:.2f}ms")

    # Dense-only
    print("\n[Dense-only]")
    dense_retriever = DenseOnlyRetriever(dense)
    results["dense_only"] = evaluate_retriever(dense_retriever, qa_items, top_k)
    print(f"  MRR: {results['dense_only']['mrr']['mean']:.4f}")
    print(f"  Recall@{top_k}: {results['dense_only']['recall@k']['mean']:.4f}")
    print(f"  Latency: {results['dense_only']['latency_ms']['mean']:.2f}ms")

    # RRF Hybrid
    print("\n[RRF Hybrid]")
    rrf_retriever = RRFHybridRetriever(dense, sparse)
    results["rrf_hybrid"] = evaluate_retriever(rrf_retriever, qa_items, top_k)
    print(f"  MRR: {results['rrf_hybrid']['mrr']['mean']:.4f}")
    print(f"  Recall@{top_k}: {results['rrf_hybrid']['recall@k']['mean']:.4f}")
    print(f"  Latency: {results['rrf_hybrid']['latency_ms']['mean']:.2f}ms")

    # 4. LightRAG 평가 (Dense + 키워드 통합)
    print("\n=== Evaluating LightRAG ===")
    print("[3/4] Initializing LightRAG with Dense retriever...")

    lightrag_dir = output_dir / "lightrag_workdir"
    try:
        from core.Services.Retrieval.LightRAG import LightRAGRetrieverWithDense

        lightrag_retriever = LightRAGRetrieverWithDense(
            working_dir=lightrag_dir,
            dense_retriever=dense,
            query_mode="hybrid",
        )
        lightrag_retriever.initialize()
        lightrag_retriever._docs = corpus  # 키워드 검색용 문서 설정

        print("\n[LightRAG + Dense]")
        results["lightrag_dense"] = evaluate_retriever(lightrag_retriever, qa_items, top_k)
        print(f"  MRR: {results['lightrag_dense']['mrr']['mean']:.4f}")
        print(f"  Recall@{top_k}: {results['lightrag_dense']['recall@k']['mean']:.4f}")
        print(f"  Latency: {results['lightrag_dense']['latency_ms']['mean']:.2f}ms")

    except Exception as e:
        print(f"  LightRAG evaluation failed: {e}")
        results["lightrag_dense"] = {"error": str(e)}

    # 결과 저장
    print("\n[4/4] Saving results...")
    summary_path = output_dir / "lightrag_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to {summary_path}")

    # 결과 요약 출력
    print("\n" + "=" * 60)
    print("SUMMARY: LightRAG vs Baselines")
    print("=" * 60)
    print(f"{'Method':<20} {'MRR':>10} {'Recall@4':>10} {'Latency':>12}")
    print("-" * 60)
    for method, data in results.items():
        if "error" not in data:
            print(f"{method:<20} {data['mrr']['mean']:>10.4f} {data['recall@k']['mean']:>10.4f} {data['latency_ms']['mean']:>10.2f}ms")
    print("=" * 60)

    # 최고 성능 확인
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_mrr = max(valid_results.items(), key=lambda x: x[1]["mrr"]["mean"])
        print(f"\nBest MRR: {best_mrr[0]} ({best_mrr[1]['mrr']['mean']:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="LightRAG vs Baseline Comparison")
    parser.add_argument("--corpus", type=Path, default=Path("output/wasabi_en_ko_parallel.jsonl"))
    parser.add_argument("--qa-file", type=Path, default=Path("output/wasabi_qa_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/experiments/lightrag"))
    parser.add_argument("--sample-size", type=int, default=50, help="QA 샘플 수 (기본: 50)")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    # 경로 조정
    base_dir = Path(__file__).parent.parent.parent.parent
    workspace_dir = base_dir.parent
    pipeline_output = workspace_dir / "smartfarm-ingest" / "output"

    corpus_path = args.corpus
    qa_path = args.qa_file

    if not corpus_path.is_absolute():
        corpus_path = pipeline_output / corpus_path.name
    if not qa_path.is_absolute():
        qa_path = pipeline_output / qa_path.name

    run_comparison(
        corpus_path=corpus_path,
        qa_path=qa_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
