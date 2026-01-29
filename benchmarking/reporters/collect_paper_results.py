#!/usr/bin/env python3
"""논문용 실험 결과 수집 및 마크다운 테이블 생성

실험 결과 JSON 파일들을 읽어 논문에 사용할 마크다운 테이블을 생성합니다.

사용법:
    python -m benchmarking.reporters.collect_paper_results \
        --results-dir output/experiments \
        --output-dir docs/paper/tables

입력 파일 구조:
    output/experiments/
    ├── baseline/
    │   └── results.json
    ├── ablation/
    │   └── results.json
    └── edge/
        └── results.json

출력 파일:
    docs/paper/tables/
    ├── table1_baseline.md
    ├── table2_ablation.md
    └── table3_edge.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """JSON 파일 로드"""
    if not filepath.exists():
        print(f"[WARN] 파일 없음: {filepath}")
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] JSON 로드 실패: {filepath} - {e}")
        return None


def format_metric(value: Optional[float], fmt: str = ".3f") -> str:
    """메트릭 값 포맷팅 (None이면 TBD)"""
    if value is None:
        return "TBD"
    return f"{value:{fmt}}"


def generate_table1_baseline(results: Optional[Dict[str, Any]], output_path: Path) -> None:
    """Table 1: Baseline Comparison (마크다운)
    
    비교 대상:
    - Dense-only
    - Sparse-only  
    - Naive Hybrid (고정 가중치)
    - HybridDAT (제안 방법)
    """
    # 결과에서 값 추출 (없으면 TBD)
    if results:
        dense = results.get("dense_only", {})
        sparse = results.get("sparse_only", {})
        naive = results.get("naive_hybrid", {})
        ours = results.get("hybriddat", {})
        env_info = results.get("environment", "TBD")
    else:
        dense = sparse = naive = ours = {}
        env_info = "TBD"
    
    # 마크다운 생성
    md = f"""# Table 1: Baseline Comparison

**설명**: Dense-only, Sparse-only, Naive Hybrid 베이스라인과 제안 방법(HybridDAT) 비교

**실험 실행**:
```bash
python -m benchmarking.experiments.baseline_comparison --corpus <corpus_path> --qa-file <qa_path>
```

**결과 파일**: `output/experiments/baseline/results.json`

---

## Retrieval Performance Comparison with Baselines

| Method | MRR@4 | Recall@4 | nDCG@4 | Latency (p95) |
|--------|-------|----------|--------|---------------|
| Dense-only | {format_metric(dense.get("mrr"))} | {format_metric(dense.get("recall"))} | {format_metric(dense.get("ndcg"))} | {format_metric(dense.get("latency_p95"), ".0f")} ms |
| Sparse-only | {format_metric(sparse.get("mrr"))} | {format_metric(sparse.get("recall"))} | {format_metric(sparse.get("ndcg"))} | {format_metric(sparse.get("latency_p95"), ".0f")} ms |
| Naive Hybrid | {format_metric(naive.get("mrr"))} | {format_metric(naive.get("recall"))} | {format_metric(naive.get("ndcg"))} | {format_metric(naive.get("latency_p95"), ".0f")} ms |
| **HybridDAT (Ours)** | **{format_metric(ours.get("mrr"))}** | **{format_metric(ours.get("recall"))}** | **{format_metric(ours.get("ndcg"))}** | {format_metric(ours.get("latency_p95"), ".0f")} ms |

---

**실험 환경**: {env_info}

---

## 메트릭 설명

- **MRR@4**: Mean Reciprocal Rank (상위 4개 결과 기준)
- **Recall@4**: 상위 4개 결과에서 정답 포함 비율
- **nDCG@4**: Normalized Discounted Cumulative Gain
- **Latency (p95)**: 95 백분위 응답 시간
"""
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"[OK] Table 1 생성: {output_path}")


def generate_table2_ablation(results: Optional[Dict[str, Any]], output_path: Path) -> None:
    """Table 2: Ablation Study (마크다운)

    컴포넌트별 기여도:
    - Full (모든 기능)
    - w/o Ontology
    - w/o PathRAG
    - w/o Dynamic Alpha
    """
    if results:
        full = results.get("full", {})
        no_onto = results.get("no_ontology", {})
        no_path = results.get("no_pathrag", {})
        no_alpha = results.get("no_dynamic_alpha", {})
        full_mrr = full.get("mrr")
    else:
        full = no_onto = no_path = no_alpha = {}
        full_mrr = None
    
    def calc_delta(val: Optional[float], base: Optional[float]) -> str:
        if val is None or base is None:
            return "TBD"
        delta = val - base
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.3f}"
    
    md = f"""# Table 2: Ablation Study

**설명**: 각 컴포넌트 제거 시 성능 변화 측정 (컴포넌트별 기여도 분석)

**실험 실행**:
```bash
python -m benchmarking.experiments.ablation_study --corpus <corpus_path> --qa-file <qa_path>
```

**결과 파일**: `output/experiments/ablation/results.json`

---

## Component Contribution Analysis

| Configuration | MRR@4 | ΔMRR | Latency (ms) |
|---------------|-------|------|--------------|
| Full (All components) | {format_metric(full_mrr)} | -- | {format_metric(full.get("latency_mean"), ".0f")} |
| w/o Ontology Matching | {format_metric(no_onto.get("mrr"))} | {calc_delta(no_onto.get("mrr"), full_mrr)} | {format_metric(no_onto.get("latency_mean"), ".0f")} |
| w/o PathRAG | {format_metric(no_path.get("mrr"))} | {calc_delta(no_path.get("mrr"), full_mrr)} | {format_metric(no_path.get("latency_mean"), ".0f")} |
| w/o Dynamic Alpha | {format_metric(no_alpha.get("mrr"))} | {calc_delta(no_alpha.get("mrr"), full_mrr)} | {format_metric(no_alpha.get("latency_mean"), ".0f")} |

---

**참고**: ΔMRR은 Full 대비 성능 변화. 음수는 성능 하락을 의미.

---

## 컴포넌트 설명

- **Ontology Matching**: 작물/환경/병해/영양소 온톨로지 기반 검색 부스팅
- **PathRAG**: 인과관계 그래프 기반 검색
- **Dynamic Alpha**: 질의 특성에 따른 Dense/Sparse 가중치 동적 조정
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"[OK] Table 2 생성: {output_path}")


def generate_table3_edge(results: Optional[Dict[str, Any]], output_path: Path) -> None:
    """Table 3: Edge Deployment Performance (마크다운)
    
    엣지 환경 성능:
    - 콜드 스타트 시간
    - 쿼리 레이턴시 (p50, p95, p99)
    - 처리량 (QPS)
    - 메모리 사용량 (peak, avg)
    """
    if results:
        latency = results.get("latency", {})
        memory = results.get("memory", {})
        startup = results.get("startup", {})
        throughput = results.get("throughput", {})
        env_info = results.get("environment", "TBD")
    else:
        latency = memory = startup = throughput = {}
        env_info = "TBD"
    
    md = f"""# Table 3: Edge Deployment Performance

**설명**: 8GB RAM 엣지 환경에서의 실시간 성능 측정

**실험 실행**:
```bash
python -m benchmarking.experiments.edge_benchmark --corpus <corpus_path> --qa-file <qa_path> --measure-memory
```

**결과 파일**: `output/experiments/edge/results.json`

---

## Edge Deployment Performance (8GB RAM)

### Startup

| Metric | Value |
|--------|-------|
| Cold Start Time | {format_metric(startup.get("cold_start_s"), ".1f")} s |
| Index Build Time | {format_metric(startup.get("index_build_s"), ".1f")} s |

### Query Latency

| Percentile | Value |
|------------|-------|
| p50 | {format_metric(latency.get("p50_ms"), ".0f")} ms |
| p95 | {format_metric(latency.get("p95_ms"), ".0f")} ms |
| p99 | {format_metric(latency.get("p99_ms"), ".0f")} ms |

### Throughput

| Metric | Value |
|--------|-------|
| Queries Per Second | {format_metric(throughput.get("qps"), ".1f")} QPS |

### Memory Usage

| Metric | Value |
|--------|-------|
| Peak RAM | {format_metric(memory.get("peak_gb"), ".2f")} GB |
| Average RAM | {format_metric(memory.get("avg_gb"), ".2f")} GB |

---

**실험 환경**: {env_info}

---

## 측정 항목 설명

- **Cold Start Time**: 서버 시작부터 첫 질의 응답 가능까지 시간
- **Index Build Time**: 전체 코퍼스 인덱싱 소요 시간
- **p50/p95/p99**: 레이턴시 백분위수 (50%, 95%, 99%)
- **QPS**: 초당 처리 가능한 질의 수
- **Peak/Average RAM**: 최대/평균 메모리 사용량
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"[OK] Table 3 생성: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="논문용 마크다운 테이블 생성")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("output/experiments"),
        help="실험 결과 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/paper/tables"),
        help="마크다운 테이블 출력 디렉토리",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("논문용 마크다운 테이블 생성")
    print("=" * 60)
    print(f"입력: {args.results_dir}")
    print(f"출력: {args.output_dir}")
    print()
    
    # 결과 파일 로드
    baseline_results = load_json(args.results_dir / "baseline" / "results.json")
    ablation_results = load_json(args.results_dir / "ablation" / "results.json")
    edge_results = load_json(args.results_dir / "edge" / "results.json")
    
    # 테이블 생성
    generate_table1_baseline(baseline_results, args.output_dir / "table1_baseline.md")
    generate_table2_ablation(ablation_results, args.output_dir / "table2_ablation.md")
    generate_table3_edge(edge_results, args.output_dir / "table3_edge.md")
    
    print()
    print("=" * 60)
    print("완료. TBD 값은 실험 실행 후 재생성하면 채워집니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
