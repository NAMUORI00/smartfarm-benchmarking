#!/usr/bin/env python3
"""
벤치마크 스모크 테스트 - 10분 이내 전체 검증

모든 벤치마킹 컴포넌트가 제대로 구현되었는지 빠르게 확인합니다.

Usage:
  EMBED_MODEL_ID=minilm python -m benchmarking.tests.test_benchmark_smoke
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarking.bootstrap import ensure_search_on_path

SEARCH_ROOT = ensure_search_on_path()
ROOT = SEARCH_ROOT

# 경량 모델 사용
os.environ.setdefault("EMBED_MODEL_ID", "minilm")

# 샘플 쿼리 (10개)
SAMPLE_QUERIES = [
    "와사비의 최적 재배 온도는?",
    "와사비 수경재배 시 적정 EC 농도는?",
    "와사비 병해충 관리 방법은?",
    "와사비 수확 시기는?",
    "와사비 재배 시 필요한 광량은?",
    "와사비 양액 pH 관리 방법은?",
    "와사비 뿌리썩음병 예방법은?",
    "와사비 육묘 기간은?",
    "와사비 정식 방법은?",
    "와사비 차광 필요성은?",
]


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None


def get_memory_mb() -> float:
    """현재 메모리 사용량 (MB)"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def test_cold_start() -> TestResult:
    """인덱스 로드 시간 측정"""
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import MiniStore
    from core.Config.Settings import settings
    
    gc.collect()
    start_mem = get_memory_mb()
    start = time.perf_counter()
    
    try:
        # Dense 인덱스 로드
        dense = EmbeddingRetriever(model_id="minilm")
        index_dir = ROOT / "data" / "index_minilm"
        
        dense_loaded = dense.load(
            str(index_dir / settings.DENSE_INDEX_FILE),
            str(index_dir / settings.DENSE_DOCS_FILE),
            mmap=True
        )
        
        # Sparse 인덱스 로드
        sparse = MiniStore()
        sparse_loaded = sparse.load(str(index_dir / settings.SPARSE_STATE_FILE))
        
        duration = (time.perf_counter() - start) * 1000
        end_mem = get_memory_mb()
        
        return TestResult(
            name="cold_start",
            passed=dense_loaded and sparse_loaded,
            duration_ms=duration,
            details={
                "dense_loaded": dense_loaded,
                "sparse_loaded": sparse_loaded,
                "n_docs": len(dense.docs),
                "memory_increase_mb": end_mem - start_mem,
            }
        )
    except Exception as e:
        return TestResult(
            name="cold_start",
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={},
            error=str(e)
        )


def test_query_latency() -> TestResult:
    """쿼리 레이턴시 측정 (10개 샘플)"""
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import MiniStore
    from core.Services.Retrieval.Hybrid import HybridDATRetriever
    from core.Config.Settings import settings
    
    start = time.perf_counter()
    
    try:
        # 인덱스 로드
        dense = EmbeddingRetriever(model_id="minilm")
        index_dir = ROOT / "data" / "index_minilm"
        dense.load(
            str(index_dir / settings.DENSE_INDEX_FILE),
            str(index_dir / settings.DENSE_DOCS_FILE),
            mmap=True
        )
        
        sparse = MiniStore()
        sparse.load(str(index_dir / settings.SPARSE_STATE_FILE))
        
        hybrid = HybridDATRetriever(dense, sparse, pathrag=None)
        
        # 쿼리 실행
        latencies = []
        for q in SAMPLE_QUERIES:
            q_start = time.perf_counter()
            results = hybrid.search(q, k=4)
            lat_ms = (time.perf_counter() - q_start) * 1000
            latencies.append(lat_ms)
        
        import numpy as np
        arr = np.array(latencies)
        
        duration = (time.perf_counter() - start) * 1000
        
        return TestResult(
            name="query_latency",
            passed=len(latencies) == len(SAMPLE_QUERIES),
            duration_ms=duration,
            details={
                "n_queries": len(SAMPLE_QUERIES),
                "first_query_ms": latencies[0],  # 모델 로딩 포함
                "mean_ms": float(arr.mean()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
            }
        )
    except Exception as e:
        return TestResult(
            name="query_latency",
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={},
            error=str(e)
        )


def test_baselines() -> TestResult:
    """Dense/Sparse/Hybrid 비교"""
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import MiniStore
    from core.Services.Retrieval.Hybrid import HybridDATRetriever
    from core.Config.Settings import settings
    
    start = time.perf_counter()
    
    try:
        # 인덱스 로드
        dense = EmbeddingRetriever(model_id="minilm")
        index_dir = ROOT / "data" / "index_minilm"
        dense.load(
            str(index_dir / settings.DENSE_INDEX_FILE),
            str(index_dir / settings.DENSE_DOCS_FILE),
            mmap=True
        )
        
        sparse = MiniStore()
        sparse.load(str(index_dir / settings.SPARSE_STATE_FILE))
        
        hybrid = HybridDATRetriever(dense, sparse, pathrag=None)
        
        # 샘플 쿼리로 각 baseline 테스트
        test_q = SAMPLE_QUERIES[0]
        
        # Dense-only
        dense_results = dense.search(test_q, k=4)
        
        # Sparse-only
        sparse_results = sparse.search(test_q, k=4)
        
        # Hybrid
        hybrid_results = hybrid.search(test_q, k=4)
        
        duration = (time.perf_counter() - start) * 1000
        
        return TestResult(
            name="baselines",
            passed=(
                len(dense_results) > 0 and
                len(sparse_results) > 0 and
                len(hybrid_results) > 0
            ),
            duration_ms=duration,
            details={
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results),
                "hybrid_results": len(hybrid_results),
                "dense_ids": [r.id for r in dense_results],
                "sparse_ids": [r.id for r in sparse_results],
                "hybrid_ids": [r.id for r in hybrid_results],
            }
        )
    except Exception as e:
        return TestResult(
            name="baselines",
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={},
            error=str(e)
        )


def test_ablation() -> TestResult:
    """컴포넌트 On/Off 테스트 (DAT)"""
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import MiniStore
    from core.Services.Retrieval.Hybrid import HybridDATRetriever
    from core.Config.Settings import settings

    start = time.perf_counter()

    try:
        # 인덱스 로드
        dense = EmbeddingRetriever(model_id="minilm")
        index_dir = ROOT / "data" / "index_minilm"
        dense.load(
            str(index_dir / settings.DENSE_INDEX_FILE),
            str(index_dir / settings.DENSE_DOCS_FILE),
            mmap=True
        )

        sparse = MiniStore()
        sparse.load(str(index_dir / settings.SPARSE_STATE_FILE))

        hybrid = HybridDATRetriever(dense, sparse, pathrag=None)

        # Dynamic Alpha Tuning 테스트
        alpha_d, alpha_s, alpha_p = hybrid.dynamic_alphas("EC 2.0 dS/m 설정")
        dat_works = alpha_s > alpha_d  # 숫자/단위 포함 시 sparse 증가

        alpha_d2, alpha_s2, alpha_p2 = hybrid.dynamic_alphas("와사비 재배")
        dat_balanced = abs(alpha_d2 - alpha_s2) < 0.3  # 일반 질의는 균형

        duration = (time.perf_counter() - start) * 1000

        return TestResult(
            name="ablation",
            passed=all([dat_works, dat_balanced]),
            duration_ms=duration,
            details={
                "dat_number_unit": {"alpha_dense": alpha_d, "alpha_sparse": alpha_s, "works": dat_works},
                "dat_balanced": {"alpha_dense": alpha_d2, "alpha_sparse": alpha_s2, "works": dat_balanced},
            }
        )
    except Exception as e:
        return TestResult(
            name="ablation",
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={},
            error=str(e)
        )


def test_cache() -> TestResult:
    """캐시 동작 테스트"""
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import MiniStore
    from core.Services.Retrieval.Hybrid import HybridDATRetriever
    from core.Config.Settings import settings
    
    start = time.perf_counter()
    
    try:
        # 인덱스 로드
        dense = EmbeddingRetriever(model_id="minilm")
        index_dir = ROOT / "data" / "index_minilm"
        dense.load(
            str(index_dir / settings.DENSE_INDEX_FILE),
            str(index_dir / settings.DENSE_DOCS_FILE),
            mmap=True
        )
        
        sparse = MiniStore()
        sparse.load(str(index_dir / settings.SPARSE_STATE_FILE))
        
        hybrid = HybridDATRetriever(dense, sparse, pathrag=None)
        
        test_q = SAMPLE_QUERIES[0]
        
        # 첫 번째 쿼리 (캐시 미스)
        t1_start = time.perf_counter()
        hybrid.search(test_q, k=4)
        t1 = (time.perf_counter() - t1_start) * 1000
        
        # 두 번째 쿼리 (캐시 히트)
        t2_start = time.perf_counter()
        hybrid.search(test_q, k=4)
        t2 = (time.perf_counter() - t2_start) * 1000
        
        # 캐시 효과: 두 번째가 훨씬 빨라야 함
        cache_effective = t2 < t1 * 0.5  # 50% 이상 빨라야 함
        
        duration = (time.perf_counter() - start) * 1000
        
        return TestResult(
            name="cache",
            passed=cache_effective,
            duration_ms=duration,
            details={
                "first_query_ms": t1,
                "second_query_ms": t2,
                "speedup": t1 / t2 if t2 > 0 else float('inf'),
                "cache_effective": cache_effective,
            }
        )
    except Exception as e:
        return TestResult(
            name="cache",
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={},
            error=str(e)
        )


def test_memory() -> TestResult:
    """메모리 사용량 측정"""
    start = time.perf_counter()
    
    try:
        mem = get_memory_mb()
        
        # 8GB 환경 기준 6GB 이하면 OK
        passed = mem < 6000
        
        return TestResult(
            name="memory",
            passed=passed,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={
                "current_mb": mem,
                "limit_mb": 6000,
                "within_limit": passed,
            }
        )
    except Exception as e:
        return TestResult(
            name="memory",
            passed=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            details={},
            error=str(e)
        )


def main():
    print("=" * 60)
    print("BENCHMARK SMOKE TEST")
    print("=" * 60)
    print(f"Target: All tests pass within 10 minutes")
    print(f"Model: {os.environ.get('EMBED_MODEL_ID', 'default')}")
    print()
    
    tests = [
        ("Cold Start", test_cold_start),
        ("Query Latency", test_query_latency),
        ("Baselines", test_baselines),
        ("Ablation", test_ablation),
        ("Cache", test_cache),
        ("Memory", test_memory),
    ]
    
    results: List[TestResult] = []
    total_start = time.perf_counter()
    
    for name, test_fn in tests:
        print(f"Running {name}...", end=" ", flush=True)
        result = test_fn()
        results.append(result)
        
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.duration_ms:.0f}ms)")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        # 주요 세부 정보 출력
        if result.name == "query_latency" and result.passed:
            print(f"  First query: {result.details.get('first_query_ms', 0):.0f}ms")
            print(f"  Mean: {result.details.get('mean_ms', 0):.0f}ms")
            print(f"  P95: {result.details.get('p95_ms', 0):.0f}ms")
        elif result.name == "cache" and result.passed:
            print(f"  Speedup: {result.details.get('speedup', 0):.1f}x")
        elif result.name == "memory" and result.passed:
            print(f"  Usage: {result.details.get('current_mb', 0):.0f}MB")
    
    total_time = time.perf_counter() - total_start
    
    # 요약
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {passed}/{len(results)} passed")
    print(f"Time: {total_time:.1f}s")
    
    if failed == 0:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print(f"\n*** {failed} TESTS FAILED ***")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error or 'check details'}")
    
    # 결과 저장
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_data = {
        "total_time_s": total_time,
        "passed": passed,
        "failed": failed,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "duration_ms": r.duration_ms,
                "details": r.details,
                "error": r.error,
            }
            for r in results
        ]
    }
    
    output_path = output_dir / "smoke_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
