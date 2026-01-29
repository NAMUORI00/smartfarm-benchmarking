#!/usr/bin/env python3
"""3회 벤치마크 실행 및 평균 계산 스크립트"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Config.Settings import settings
from benchmarking.experiments.edge_benchmark import EdgeBenchmark

def run_benchmarks(n_iterations: int = 3):
    """벤치마크를 n번 실행하고 결과 수집"""

    # 경로 설정
    corpus_path = Path(settings.get_corpus_path())
    qa_path = Path(settings.get_qa_dataset_path())
    output_dir = Path(__file__).parent / "output" / "benchmark_3x"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus: {corpus_path}")
    print(f"QA: {qa_path}")
    print(f"Output: {output_dir}")

    all_results = []

    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"BENCHMARK RUN {i+1}/{n_iterations}")
        print(f"{'='*60}")

        run_output_dir = output_dir / f"run_{i+1}"

        benchmark = EdgeBenchmark(
            corpus_path=corpus_path,
            qa_path=qa_path,
            output_dir=run_output_dir,
            top_k=4,
            warmup_queries=10,
            n_runs=3,
        )

        results = benchmark.run_full_benchmark()
        all_results.append(results)

    # 평균 계산
    print("\n" + "="*60)
    print("CALCULATING AVERAGES")
    print("="*60)

    # 메트릭 수집
    cold_start_times = [r["cold_start"]["total_cold_start_time_s"] for r in all_results]
    index_memories = [r["cold_start"]["total_memory_increase_mb"] for r in all_results]
    latency_p50s = [r["query_latency"]["latency_ms"]["p50"] for r in all_results]
    latency_p95s = [r["query_latency"]["latency_ms"]["p95"] for r in all_results]
    latency_p99s = [r["query_latency"]["latency_ms"]["p99"] for r in all_results]
    qps_values = [r["query_latency"]["qps"] for r in all_results]

    # 평균
    avg_cold_start = sum(cold_start_times) / len(cold_start_times)
    avg_index_memory = sum(index_memories) / len(index_memories)
    avg_p50 = sum(latency_p50s) / len(latency_p50s)
    avg_p95 = sum(latency_p95s) / len(latency_p95s)
    avg_p99 = sum(latency_p99s) / len(latency_p99s)
    avg_qps = sum(qps_values) / len(qps_values)

    # 목표치 비교
    targets = {
        "cold_start_time_s": {"target": 10, "op": "<"},
        "index_memory_mb": {"target": 1024, "op": "<"},
        "latency_p50_ms": {"target": 200, "op": "<"},
        "latency_p95_ms": {"target": 500, "op": "<"},
        "latency_p99_ms": {"target": 1000, "op": "<"},
        "qps": {"target": 5, "op": ">"},
    }

    def check_pass(value, target, op):
        if op == "<":
            return value < target
        elif op == ">":
            return value > target
        return False

    summary = {
        "n_iterations": n_iterations,
        "individual_runs": {
            "cold_start_time_s": cold_start_times,
            "index_memory_mb": index_memories,
            "latency_p50_ms": latency_p50s,
            "latency_p95_ms": latency_p95s,
            "latency_p99_ms": latency_p99s,
            "qps": qps_values,
        },
        "averages": {
            "cold_start_time_s": avg_cold_start,
            "index_memory_mb": avg_index_memory,
            "latency_p50_ms": avg_p50,
            "latency_p95_ms": avg_p95,
            "latency_p99_ms": avg_p99,
            "qps": avg_qps,
        },
        "targets": {
            "cold_start_time_s": {"target": "< 10s", "pass": check_pass(avg_cold_start, 10, "<")},
            "index_memory_mb": {"target": "< 1GB", "pass": check_pass(avg_index_memory, 1024, "<")},
            "latency_p50_ms": {"target": "< 200ms", "pass": check_pass(avg_p50, 200, "<")},
            "latency_p95_ms": {"target": "< 500ms", "pass": check_pass(avg_p95, 500, "<")},
            "latency_p99_ms": {"target": "< 1000ms", "pass": check_pass(avg_p99, 1000, "<")},
            "qps": {"target": "> 5 QPS", "pass": check_pass(avg_qps, 5, ">")},
        },
    }

    # 결과 저장
    summary_path = output_dir / "benchmark_3x_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    print("\n" + "="*80)
    print("FINAL RESULTS (3-RUN AVERAGE)")
    print("="*80)
    print(f"\n{'Metric':<25} {'Value':<15} {'Target':<15} {'Status':<10}")
    print("-"*70)

    metrics = [
        ("Cold Start Time", f"{avg_cold_start:.2f} s", "< 10s", summary["targets"]["cold_start_time_s"]["pass"]),
        ("Index Memory", f"{avg_index_memory:.1f} MB", "< 1GB", summary["targets"]["index_memory_mb"]["pass"]),
        ("Query Latency (p50)", f"{avg_p50:.1f} ms", "< 200ms", summary["targets"]["latency_p50_ms"]["pass"]),
        ("Query Latency (p95)", f"{avg_p95:.1f} ms", "< 500ms", summary["targets"]["latency_p95_ms"]["pass"]),
        ("Query Latency (p99)", f"{avg_p99:.1f} ms", "< 1000ms", summary["targets"]["latency_p99_ms"]["pass"]),
        ("Throughput", f"{avg_qps:.1f} QPS", "> 5", summary["targets"]["qps"]["pass"]),
    ]

    all_pass = True
    for name, value, target, passed in metrics:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"{name:<25} {value:<15} {target:<15} {status:<10}")

    print("-"*70)
    overall = "ALL TARGETS MET" if all_pass else "SOME TARGETS FAILED"
    print(f"\nOverall: {overall}")
    print(f"\nDetailed results saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    run_benchmarks(3)
