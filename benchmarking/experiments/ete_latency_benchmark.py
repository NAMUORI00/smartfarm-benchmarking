#!/usr/bin/env python3
"""End-to-End Latency Benchmark

검색(Retrieval) + LLM 추론(Generation)까지의 전체 레이턴시 측정.

측정 항목:
- Retrieval Latency: 검색 레이턴시 (p50, p95, p99)
- Generation Latency: LLM 추론 레이턴시 (p50, p95, p99)
- End-to-End Latency: 전체 레이턴시 (p50, p95, p99)
- Throughput: 초당 처리 가능한 질의 수 (QPS)

사용 예시:
    python -m benchmarking.experiments.ete_latency_benchmark \
        --corpus /path/to/corpus.jsonl \
        --qa-file /path/to/qa_dataset.jsonl \
        --output-dir output/experiments/ete \
        --n-samples 50

    # Or set DATA_ROOT env var to use defaults from Settings

"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store, MiniStore
from core.Services.Retrieval.Hybrid import HybridDATRetriever
from core.Services.LLM import generate_json
from core.Services.PromptTemplates import build_rag_answer_prompt
from core.Config.Settings import settings
from core.Models.Schemas import SourceDoc


@dataclass
class QAItem:
    """QA 데이터셋 항목"""
    id: str
    question: str


@dataclass
class LatencyStats:
    """레이턴시 통계"""
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": round(self.mean, 2),
            "std_ms": round(self.std, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "p50_ms": round(self.p50, 2),
            "p75_ms": round(self.p75, 2),
            "p90_ms": round(self.p90, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
        }


def load_qa_dataset(path: Path, limit: Optional[int] = None) -> List[QAItem]:
    """QA 데이터셋에서 질문만 로드"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(QAItem(
                id=data["id"],
                question=data["question"],
            ))
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


def compute_latency_stats(latencies: List[float]) -> LatencyStats:
    """레이턴시 통계 계산"""
    arr = np.array(latencies)
    return LatencyStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


class EtELatencyBenchmark:
    """End-to-End 레이턴시 벤치마크"""

    def __init__(
        self,
        corpus_path: Path,
        qa_path: Path,
        output_dir: Path,
        top_k: int = 4,
        warmup_queries: int = 3,
        n_samples: int = 50,
    ):
        self.corpus_path = corpus_path
        self.qa_path = qa_path
        self.output_dir = output_dir
        self.top_k = top_k
        self.warmup_queries = warmup_queries
        self.n_samples = n_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        """리트리버 초기화"""
        print("\n=== Setting up retrievers ===")

        # 말뭉치 로드
        print("  Loading corpus...")
        self.corpus = load_corpus(self.corpus_path)
        print(f"  Loaded {len(self.corpus)} documents")

        # Dense 인덱스 빌드
        print("  Building Dense index...")
        self.dense = EmbeddingRetriever()
        self.dense.build(self.corpus)

        # Sparse 인덱스 빌드
        sparse_method = getattr(settings, "SPARSE_METHOD", "bm25").lower()
        print(f"  Building Sparse index (method={sparse_method})...")
        self.sparse = MiniStore() if sparse_method == "tfidf" else BM25Store()
        self.sparse.index(self.corpus)

        # 하이브리드 리트리버
        self.retriever = HybridDATRetriever(
            dense=self.dense,
            sparse=self.sparse,
            pathrag=None,
        )

        print("  Setup complete!")

    def benchmark_ete_latency(self) -> Dict[str, Any]:
        """End-to-End 레이턴시 벤치마크"""
        print("\n=== End-to-End Latency Benchmark ===")

        # QA 로드
        qa_items = load_qa_dataset(self.qa_path, limit=self.n_samples)
        print(f"  Loaded {len(qa_items)} queries (limit: {self.n_samples})")

        # Warmup
        print(f"  Warming up with {self.warmup_queries} queries...")
        for qa in qa_items[:self.warmup_queries]:
            # 검색
            hits = self.retriever.search(qa.question, k=self.top_k)
            # LLM 생성
            if hits:
                prompt = build_rag_answer_prompt(qa.question, hits)
                try:
                    generate_json(prompt)
                except Exception as e:
                    print(f"    Warmup LLM error: {e}")

        # 실제 벤치마크
        retrieval_latencies: List[float] = []
        generation_latencies: List[float] = []
        ete_latencies: List[float] = []
        failed_count = 0

        print(f"  Running benchmark on {len(qa_items)} queries...")
        for i, qa in enumerate(qa_items):
            ete_start = time.perf_counter()

            # 1. 검색
            retrieval_start = time.perf_counter()
            hits = self.retriever.search(qa.question, k=self.top_k)
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
            retrieval_latencies.append(retrieval_time_ms)

            # 2. LLM 생성
            generation_start = time.perf_counter()
            if hits:
                prompt = build_rag_answer_prompt(qa.question, hits)
                try:
                    generate_json(prompt)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        print(f"    LLM error on query {i}: {e}")
            generation_time_ms = (time.perf_counter() - generation_start) * 1000
            generation_latencies.append(generation_time_ms)

            # 3. 전체 시간
            ete_time_ms = (time.perf_counter() - ete_start) * 1000
            ete_latencies.append(ete_time_ms)

            # 진행 상황 출력
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(qa_items)} queries")

        # 통계 계산
        retrieval_stats = compute_latency_stats(retrieval_latencies)
        generation_stats = compute_latency_stats(generation_latencies)
        ete_stats = compute_latency_stats(ete_latencies)

        # QPS 계산
        total_time_s = sum(ete_latencies) / 1000
        qps = len(qa_items) / total_time_s if total_time_s > 0 else 0

        result = {
            "n_queries": len(qa_items),
            "n_failed": failed_count,
            "retrieval_latency": retrieval_stats.to_dict(),
            "generation_latency": generation_stats.to_dict(),
            "ete_latency": ete_stats.to_dict(),
            "qps": round(qps, 2),
        }

        # 결과 출력
        print("\n  === Results ===")
        print(f"  Queries: {len(qa_items)} (failed: {failed_count})")
        print(f"\n  Retrieval Latency:")
        print(f"    p50: {retrieval_stats.p50:.2f}ms, p95: {retrieval_stats.p95:.2f}ms, p99: {retrieval_stats.p99:.2f}ms")
        print(f"\n  Generation (LLM) Latency:")
        print(f"    p50: {generation_stats.p50:.2f}ms, p95: {generation_stats.p95:.2f}ms, p99: {generation_stats.p99:.2f}ms")
        print(f"\n  End-to-End Latency:")
        print(f"    p50: {ete_stats.p50:.2f}ms, p95: {ete_stats.p95:.2f}ms, p99: {ete_stats.p99:.2f}ms")
        print(f"\n  Throughput: {qps:.2f} QPS")

        return result

    def run(self) -> Dict[str, Any]:
        """전체 벤치마크 실행"""
        print("=" * 60)
        print("END-TO-END LATENCY BENCHMARK")
        print("=" * 60)

        results = {
            "config": {
                "corpus_path": str(self.corpus_path),
                "qa_path": str(self.qa_path),
                "top_k": self.top_k,
                "warmup_queries": self.warmup_queries,
                "n_samples": self.n_samples,
            },
            "environment": {
                "python_version": sys.version.split()[0],
            },
        }

        # 셋업
        self.setup()

        # 벤치마크 실행
        results["benchmark"] = self.benchmark_ete_latency()

        # 결과 저장
        results_path = self.output_dir / "ete_latency_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[EtEBenchmark] Results saved to {results_path}")

        # 요약 저장
        summary = {
            "retrieval_p50_ms": results["benchmark"]["retrieval_latency"]["p50_ms"],
            "retrieval_p95_ms": results["benchmark"]["retrieval_latency"]["p95_ms"],
            "retrieval_p99_ms": results["benchmark"]["retrieval_latency"]["p99_ms"],
            "generation_p50_ms": results["benchmark"]["generation_latency"]["p50_ms"],
            "generation_p95_ms": results["benchmark"]["generation_latency"]["p95_ms"],
            "generation_p99_ms": results["benchmark"]["generation_latency"]["p99_ms"],
            "ete_p50_ms": results["benchmark"]["ete_latency"]["p50_ms"],
            "ete_p95_ms": results["benchmark"]["ete_latency"]["p95_ms"],
            "ete_p99_ms": results["benchmark"]["ete_latency"]["p99_ms"],
            "qps": results["benchmark"]["qps"],
        }

        summary_path = self.output_dir / "ete_latency_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[EtEBenchmark] Summary saved to {summary_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="End-to-End Latency Benchmark")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="말뭉치 JSONL 파일 경로 (default: from DATA_ROOT or CORPUS_PATH env)",
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=None,
        help="QA 데이터셋 JSONL 파일 경로 (default: from DATA_ROOT or QA_DATASET_PATH env)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/ete"),
        help="결과 출력 디렉토리",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="검색 결과 수 (기본: 4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="웜업 쿼리 수 (기본: 3)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="벤치마크할 샘플 수 (기본: 50)",
    )
    args = parser.parse_args()

    # Fallback to Settings defaults if not provided
    if args.corpus is None:
        args.corpus = Path(settings.get_corpus_path())
    if args.qa_file is None:
        args.qa_file = Path(settings.get_qa_dataset_path())

    benchmark = EtELatencyBenchmark(
        corpus_path=args.corpus,
        qa_path=args.qa_file,
        output_dir=args.output_dir,
        top_k=args.top_k,
        warmup_queries=args.warmup,
        n_samples=args.n_samples,
    )

    benchmark.run()


if __name__ == "__main__":
    main()
