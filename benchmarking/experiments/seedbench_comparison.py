#!/usr/bin/env python3
"""SeedBench 기반 Sparse vs LightRAG 비교 실험

ACL 2025 SeedBench 데이터셋을 사용하여 Sparse-only와 LightRAG의
검색 품질을 비교합니다.

사용 예시:
    python -m benchmarking.experiments.seedbench_comparison \
        --corpus-path ../SeedBench/corpus/279segments.json \
        --qa-path ../SeedBench/base_model_eval/qa_base.json \
        --output-dir output/experiments/seedbench
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault('LLMLITE_HOST', 'http://localhost:45857')

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Sparse import MiniStore
from core.Models.Schemas import SourceDoc

# Dense retrieval
from core.Services.Retrieval.Embeddings import EmbeddingRetriever

# Baselines
from benchmarking.baselines import (
    DenseOnlyRetriever,
    RRFHybridRetriever,
    AdaptiveHybridRetriever,
)

# LightRAG
try:
    from core.Services.Retrieval.LightRAG import LightRAGRetriever
    HAS_LIGHTRAG = True
except ImportError:
    HAS_LIGHTRAG = False
    print("[WARNING] LightRAG not available")

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SEEDBENCH_CORPUS = _WORKSPACE_ROOT / "SeedBench" / "corpus" / "279segments.json"
_DEFAULT_SEEDBENCH_QA = _WORKSPACE_ROOT / "SeedBench" / "base_model_eval" / "qa_base.json"


@dataclass
class QAItem:
    """SeedBench QA 항목"""
    id: str
    question: str
    answer: str
    task_type: str = "fill_blank"
    level1: str = ""
    level2: str = ""


def load_seedbench_corpus(path: Path, limit: Optional[int] = None) -> List[SourceDoc]:
    """SeedBench 말뭉치 로드 (279segments.json - JSONL 형식)"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            # segment 필드가 텍스트
            text = item.get("segment", "")
            subcategory = item.get("subcategory", "")

            if not text:
                continue

            docs.append(SourceDoc(
                id=f"seedbench_{i}",
                text=text,
                metadata={"subcategory": subcategory, "source": "seedbench"}
            ))

            if limit and len(docs) >= limit:
                break

    return docs


def load_seedbench_qa(path: Path, limit: Optional[int] = None) -> List[QAItem]:
    """SeedBench QA 데이터셋 로드"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        question = item.get("question", "")
        answer = item.get("answer", "")

        # MCQ의 경우 answer가 A, B, C, D 등의 문자
        # Fill-blank의 경우 실제 답변

        # split 정보 (있는 경우)
        split_info = item.get("split", {})
        level1 = split_info.get("level1", "")
        level2 = split_info.get("level2", "")
        task_type = item.get("task_type", "fill_blank")

        if not question:
            continue

        items.append(QAItem(
            id=f"qa_{i}",
            question=question,
            answer=answer,
            task_type=task_type,
            level1=level1,
            level2=level2,
        ))

        if limit and len(items) >= limit:
            break

    return items


def compute_context_relevance_simple(question: str, contexts: List[str]) -> float:
    """간단한 컨텍스트 관련성 계산 (키워드 기반)

    LLM 호출 없이 키워드 겹침으로 관련성 추정
    """
    if not contexts:
        return 0.0

    # 질문에서 키워드 추출 (불용어 제거)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "how", "to", "of", "in", "on", "for", "with", "by", "from", "at", "as", "that", "this", "it", "and", "or", "but", "not", "be", "have", "has", "had", "do", "does", "did"}
    q_words = set(w.lower() for w in question.split() if len(w) > 2 and w.lower() not in stopwords)

    # 각 컨텍스트와 겹침 계산
    scores = []
    for ctx in contexts:
        ctx_words = set(w.lower() for w in ctx.split() if len(w) > 2)
        overlap = len(q_words & ctx_words)
        score = overlap / max(len(q_words), 1)
        scores.append(min(score, 1.0))

    return max(scores) if scores else 0.0


def compute_answer_coverage(answer: str, contexts: List[str]) -> float:
    """답변이 컨텍스트에 포함되어 있는지 확인"""
    if not contexts or not answer:
        return 0.0

    answer_lower = answer.lower()
    for ctx in contexts:
        if answer_lower in ctx.lower():
            return 1.0

    # 부분 매칭
    answer_words = set(answer_lower.split())
    for ctx in contexts:
        ctx_lower = ctx.lower()
        matched = sum(1 for w in answer_words if w in ctx_lower)
        if matched >= len(answer_words) * 0.5:
            return 0.5

    return 0.0


def evaluate_retriever(
    retriever,
    qa_items: List[QAItem],
    top_k: int = 4,
    name: str = "",
) -> Dict[str, Any]:
    """리트리버 평가 (keyword 기반 관련성)"""
    latencies = []
    relevance_scores = []
    answer_coverage_scores = []

    for i, qa in enumerate(qa_items):
        start = time.perf_counter()
        results = retriever.search(qa.question, k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        contexts = [r.text for r in results]

        # 관련성 점수 (키워드 기반)
        relevance = compute_context_relevance_simple(qa.question, contexts)
        relevance_scores.append(relevance)

        # 답변 포함 여부
        coverage = compute_answer_coverage(qa.answer, contexts)
        answer_coverage_scores.append(coverage)

        if (i + 1) % 20 == 0:
            print(f"  [{name}] Processed {i + 1}/{len(qa_items)} queries")

    return {
        "n_queries": len(qa_items),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "context_relevance": {
            "mean": float(np.mean(relevance_scores)),
            "std": float(np.std(relevance_scores)),
        },
        "answer_coverage": {
            "mean": float(np.mean(answer_coverage_scores)),
            "std": float(np.std(answer_coverage_scores)),
        },
    }


class SparseOnlyRetriever:
    """Sparse-only 베이스라인"""
    def __init__(self, sparse: MiniStore):
        self.sparse = sparse

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        return self.sparse.search(q, k=k)


def main():
    parser = argparse.ArgumentParser(description="SeedBench Comparison")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=_DEFAULT_SEEDBENCH_CORPUS,
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=_DEFAULT_SEEDBENCH_QA,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/experiments/seedbench"))
    parser.add_argument("--doc-limit", type=int, default=None, help="문서 수 제한")
    parser.add_argument("--qa-limit", type=int, default=100, help="QA 수 제한")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--skip-lightrag", action="store_true", help="LightRAG 스킵")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lightrag_dir = args.output_dir / "lightrag_workdir"

    # Clean previous workdir
    if lightrag_dir.exists():
        shutil.rmtree(lightrag_dir)

    # 1. 데이터 로드
    print(f"[1/5] Loading SeedBench corpus from {args.corpus_path}")
    corpus = load_seedbench_corpus(args.corpus_path, limit=args.doc_limit)
    print(f"  Loaded {len(corpus)} documents")

    print(f"[2/5] Loading SeedBench QA from {args.qa_path} (limit={args.qa_limit})")
    qa_items = load_seedbench_qa(args.qa_path, limit=args.qa_limit)
    print(f"  Loaded {len(qa_items)} QA items")

    # 샘플 출력
    print("\n--- Sample QA ---")
    for qa in qa_items[:3]:
        print(f"Q: {qa.question[:80]}...")
        print(f"A: {qa.answer}")
        print()

    # 2. Sparse 인덱스 구축
    print("[3/5] Building Sparse index...")
    sparse = MiniStore()
    sparse.index(corpus)
    sparse_retriever = SparseOnlyRetriever(sparse)

    # Dense 인덱스 구축
    print("  Building Dense index...")
    dense = EmbeddingRetriever()
    dense.build(corpus)

    results = {}

    # 3. Sparse-only 평가
    print("\n=== Evaluating Sparse-only ===")
    results["sparse_only"] = evaluate_retriever(
        sparse_retriever, qa_items, args.top_k, "Sparse"
    )
    print(f"  Context Relevance: {results['sparse_only']['context_relevance']['mean']:.4f}")
    print(f"  Answer Coverage: {results['sparse_only']['answer_coverage']['mean']:.4f}")
    print(f"  Latency: {results['sparse_only']['latency_ms']['mean']:.2f}ms")

    # 4. Dense-only 평가
    print("\n=== Evaluating Dense-only ===")
    dense_retriever = DenseOnlyRetriever(dense)
    results["dense_only"] = evaluate_retriever(
        dense_retriever, qa_items, args.top_k, "Dense"
    )
    print(f"  Context Relevance: {results['dense_only']['context_relevance']['mean']:.4f}")
    print(f"  Answer Coverage: {results['dense_only']['answer_coverage']['mean']:.4f}")
    print(f"  Latency: {results['dense_only']['latency_ms']['mean']:.2f}ms")

    # 5. RRF Hybrid 평가
    print("\n=== Evaluating RRF Hybrid ===")
    rrf_retriever = RRFHybridRetriever(dense, sparse)
    results["rrf_hybrid"] = evaluate_retriever(
        rrf_retriever, qa_items, args.top_k, "RRF-Hybrid"
    )
    print(f"  Context Relevance: {results['rrf_hybrid']['context_relevance']['mean']:.4f}")
    print(f"  Answer Coverage: {results['rrf_hybrid']['answer_coverage']['mean']:.4f}")
    print(f"  Latency: {results['rrf_hybrid']['latency_ms']['mean']:.2f}ms")

    # 6. Adaptive Hybrid 평가
    print("\n=== Evaluating Adaptive Hybrid ===")
    adaptive_retriever = AdaptiveHybridRetriever(dense, sparse)
    results["adaptive_hybrid"] = evaluate_retriever(
        adaptive_retriever, qa_items, args.top_k, "Adaptive-Hybrid"
    )
    print(f"  Context Relevance: {results['adaptive_hybrid']['context_relevance']['mean']:.4f}")
    print(f"  Answer Coverage: {results['adaptive_hybrid']['answer_coverage']['mean']:.4f}")
    print(f"  Latency: {results['adaptive_hybrid']['latency_ms']['mean']:.2f}ms")
    # Adaptive routing 통계
    routing_stats = adaptive_retriever.get_routing_stats()
    print(f"  Routing: {routing_stats}")

    # 7. LightRAG 구축 및 평가
    if not args.skip_lightrag and HAS_LIGHTRAG:
        print(f"\n[7/8] Building LightRAG graph ({len(corpus)} documents)...")
        print("  This will take time due to LLM-based entity extraction...")

        try:
            lightrag = LightRAGRetriever(
                working_dir=lightrag_dir,
                query_mode="hybrid",
            )
            lightrag.initialize()

            # 배치 삽입
            start_time = time.time()
            print(f"  Inserting {len(corpus)} documents...")
            lightrag.insert_docs(corpus)
            build_time = time.time() - start_time
            print(f"  Graph built in {build_time/60:.1f} minutes")

            print("\n=== Evaluating LightRAG ===")
            results["lightrag"] = evaluate_retriever(
                lightrag, qa_items, args.top_k, "LightRAG"
            )
            results["lightrag"]["build_time_sec"] = build_time
            print(f"  Context Relevance: {results['lightrag']['context_relevance']['mean']:.4f}")
            print(f"  Answer Coverage: {results['lightrag']['answer_coverage']['mean']:.4f}")
            print(f"  Latency: {results['lightrag']['latency_ms']['mean']:.2f}ms")

        except Exception as e:
            print(f"  LightRAG failed: {e}")
            results["lightrag"] = {"error": str(e)}
    else:
        print("\n[7/8] Skipping LightRAG")
        results["lightrag"] = {"skipped": True}

    # 8. 결과 저장
    print("\n[8/8] Saving results...")

    results["metadata"] = {
        "corpus_size": len(corpus),
        "qa_size": len(qa_items),
        "top_k": args.top_k,
        "corpus_path": str(args.corpus_path),
        "qa_path": str(args.qa_path),
    }

    result_path = args.output_dir / "seedbench_comparison.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to {result_path}")

    # 요약 출력
    print("\n" + "=" * 70)
    print("SUMMARY: SeedBench Sparse vs LightRAG")
    print("=" * 70)
    print(f"Corpus: {len(corpus)} documents")
    print(f"QA: {len(qa_items)} questions")
    print("-" * 70)
    print(f"{'Method':<20} {'Relevance':>12} {'Coverage':>12} {'Latency':>12}")
    print("-" * 70)

    for method in ["sparse_only", "dense_only", "rrf_hybrid", "adaptive_hybrid", "lightrag"]:
        data = results.get(method, {})
        if "error" in data or "skipped" in data:
            print(f"{method:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        else:
            rel = data.get("context_relevance", {}).get("mean", 0)
            cov = data.get("answer_coverage", {}).get("mean", 0)
            lat = data.get("latency_ms", {}).get("mean", 0)
            print(f"{method:<20} {rel:>12.4f} {cov:>12.4f} {lat:>10.2f}ms")

    print("=" * 70)

    # 승자 판정
    sparse_rel = results.get("sparse_only", {}).get("context_relevance", {}).get("mean", 0)
    lightrag_data = results.get("lightrag", {})

    if "error" not in lightrag_data and "skipped" not in lightrag_data:
        lightrag_rel = lightrag_data.get("context_relevance", {}).get("mean", 0)

        if lightrag_rel > sparse_rel:
            improvement = ((lightrag_rel / max(sparse_rel, 0.001)) - 1) * 100
            print(f"\n[WINNER] LightRAG wins! (+{improvement:.1f}% relevance)")
        else:
            if sparse_rel > 0:
                decline = ((sparse_rel / max(lightrag_rel, 0.001)) - 1) * 100
                print(f"\n[RESULT] Sparse-only still better ({decline:.1f}% higher relevance)")
            else:
                print(f"\n[RESULT] Both methods have 0 relevance")


if __name__ == "__main__":
    main()
