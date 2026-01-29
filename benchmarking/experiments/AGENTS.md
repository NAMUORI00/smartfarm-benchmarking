<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# experiments/

RAG 평가 및 벤치마크 실험 스크립트 모음.

## Evaluation Framework (6-Layer)

본 프로젝트는 Graph RAG 시스템의 종합적 평가를 위한 6-Layer 평가 프레임워크를 사용합니다.

### Research Questions (RQ)

> **서론 1.6절 연구 목표와의 대응:**
> - 목표 (1): 현장 디바이스에서 동작 가능한 경량 LLM 추론 환경 구축 → RQ1
> - 목표 (2): 매뉴얼·가이드 등을 참조 지식으로 정리하여 근거 기반 응답 지원 → RQ2
> - 목표 (3): 질의 유형에 따라 검색 및 컨텍스트 조절로 품질-비용 균형 → RQ3
> - 목표 (4): 응답 시간, 메모리, 정확도 등 지표를 통한 성능 평가 → RQ4

- **RQ1**: 엣지 성능 평가 - 엣지 환경(8GB RAM)에서 동작 가능한가?
  - 평가 스크립트: `ete_latency_benchmark.py`
  - Cold Start Time, Query Latency (p50/p95/p99), Memory Usage, QPS 측정

- **RQ2**: 검색 및 생성 품질 평가 - 근거 기반 응답을 지원하는가?
  - 평가 스크립트: BEIR 벤치마크 (검색), `ragas_eval.py` (생성 품질)
  - IR 메트릭 (MRR, NDCG@K) + RAGAS 메트릭 (Faithfulness, Answer Relevance)

- **RQ3**: Ablation Study - 품질-비용 균형을 위한 컴포넌트 기여도는?
  - 평가 스크립트: `llm_graph_ab_test.py`, ablation 실험
  - RRF, DAT, Ontology, PathRAG 각 컴포넌트별 기여도 분석

- **RQ4**: 도메인 분석 - 도메인 특화 기능이 효과적인가?
  - 평가 스크립트: `causal_extraction_eval.py`, `multihop_eval.py`
  - 카테고리별/복잡도별 성능, 인과관계 추출 품질, Multi-hop 추론 정확도

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | 패키지 초기화 |
| `ete_latency_benchmark.py` | **RQ1**: 엣지 성능 평가 (latency, memory, QPS) |
| `ragas_eval.py` | **RQ2**: 생성 품질 평가 (faithfulness, relevance 등) |
| `llm_graph_ab_test.py` | **RQ3**: 그래프 빌드 모드 비교 (rule/llm/hybrid) |
| `causal_extraction_eval.py` | **RQ4**: 인과관계 추출 품질 평가 (Entity/Relation P/R/F1) |
| `multihop_eval.py` | **RQ4**: Multi-hop 추론 평가 (Hop accuracy, Path match) |
| `crop_pathrag_eval.py` | CROP 데이터셋 PathRAG 평가 |
| `crop_lightrag_eval.py` | CROP 데이터셋 LightRAG 평가 |
| `pathrag_ragas_eval.py` | PathRAG RAGAS 평가 (RQ4) |
| `batch_eval_rag.py` | 배치 쿼리 평가 (실행 중인 서버 대상) |
| `beir_benchmark.py` | **RQ2**: BEIR 표준 벤치마크 평가 (⚠️ Reference Only) |
| `ete_latency_benchmark.py` | End-to-end 레이턴시 벤치마크 |
| `eval_chunking_configs.py` | 청킹 설정 평가 |
| `hybrid_pathrag_lt_benchmark.py` | Hybrid PathRAG local-tree 벤치마크 |
| `lightrag_comparison.py` | LightRAG 비교 실험 |
| `lightrag_graph_test.py` | LightRAG 그래프 테스트 |
| `llm_graph_benchmark.py` | LLM 그래프 벤치마크 |
| `pathrag_lt_benchmark.py` | PathRAG local-tree 벤치마크 |
| `RagExperimentRunner.py` | 다중 설정 실험 러너 |
| `seed_strategy_benchmark.py` | Seed 전략 벤치마크 |
| `seedbench_comparison.py` | Seed 벤치마크 비교 |

## 실험 카테고리

### Primary Evaluation (Graph RAG)
RQ 평가를 위한 핵심 실험 스크립트.

**RQ1 - 엣지 성능:**
- `ete_latency_benchmark.py`: End-to-end 레이턴시 및 리소스 사용량 측정
  - Cold Start Time, Query Latency (p50/p95/p99), Memory Usage, QPS

**RQ2 - 검색 및 생성 품질:**
- `ragas_eval.py`: RAGAS 메트릭 종합 평가 (faithfulness, answer relevance, context precision/recall)
- `pathrag_ragas_eval.py`: PathRAG 전용 RAGAS 평가
- BEIR 벤치마크: 검색 성능 평가 (MRR, NDCG@K)

**RQ3 - Ablation Study:**
- `llm_graph_ab_test.py`: 그래프 빌드 모드 비교 (rule_only, llm_only, hybrid)
- Ablation 실험: RRF, DAT, Ontology, PathRAG 컴포넌트별 기여도

**RQ4 - 도메인 분석:**
- `causal_extraction_eval.py`: Entity/Relation Precision, Recall, F1 측정
- `multihop_eval.py`: Hop Accuracy, Path Match Rate 측정

### Baseline Comparisons
제안 방법과 baseline 시스템 비교 실험.

**PathRAG 실험:**
- `pathrag_lt_benchmark.py`: PathRAG local-tree 성능 평가
- `hybrid_pathrag_lt_benchmark.py`: Hybrid PathRAG 전략 성능 비교
- `crop_pathrag_eval.py`: CROP 데이터셋 PathRAG 평가

**LightRAG 실험:**
- `lightrag_comparison.py`: 제안 방법 vs LightRAG 비교
- `lightrag_graph_test.py`: LightRAG 그래프 구조 테스트
- `crop_lightrag_eval.py`: CROP 데이터셋 LightRAG 평가
- `llm_graph_benchmark.py`: LLM 기반 그래프 벤치마크

**Seed 전략 실험:**
- `seed_strategy_benchmark.py`: 다양한 seed 전략 비교
- `seedbench_comparison.py`: Seed 벤치마크 결과 비교

### Reference Only (Retrieval)
⚠️ **REFERENCE ONLY**: BEIR evaluates retrieval metrics only (MRR, NDCG).
Complements RQ2 (검색 품질) but does NOT measure Graph RAG-specific capabilities.

**BEIR 벤치마크 (RQ2 보완):**
- `beir_benchmark.py`: BEIR 데이터셋으로 retrieval 성능 측정
  - MRR, NDCG@K, Precision@K, Recall@K 산출
  - RQ2의 검색 성능 부분을 외부 벤치마크로 보완

### Utilities
범용 평가 도구 및 성능 측정.

**평가 도구:**
- `RagExperimentRunner.py`: 다중 설정 실험을 위한 러너 클래스
- `batch_eval_rag.py`: 배치 단위 쿼리 평가
- `eval_chunking_configs.py`: 청킹 전략별 성능 평가
- `ete_latency_benchmark.py`: End-to-end 레이턴시 측정

## For AI Agents

### 실험 실행 예시

#### Primary Evaluation (RQ Framework)

**RQ1 - 엣지 성능 평가:**
```bash
# End-to-end 레이턴시 및 리소스 측정
python -m benchmarking.experiments.ete_latency_benchmark \
    --iterations 100 \
    --output-dir output/rq1_edge
```

**RQ2 - 검색 및 생성 품질 평가:**
```bash
# RAGAS 메트릭 종합 평가
python -m benchmarking.experiments.ragas_eval \
    --results results.jsonl \
    --output-dir output/rq2_retrieval_quality

# PathRAG RAGAS 평가
python -m benchmarking.experiments.pathrag_ragas_eval \
    --results pathrag_results.jsonl \
    --output-dir output/rq2_pathrag
```

**RQ3 - Ablation Study:**
```bash
# 그래프 빌드 모드 비교
python -m benchmarking.experiments.llm_graph_ab_test \
    --corpus path/to/corpus.jsonl \
    --output-dir output/rq3_ablation
```

**RQ4 - 도메인 분석:**
```bash
# 인과관계 추출 품질 평가
python -m benchmarking.experiments.causal_extraction_eval \
    --gold-file data/causal_extraction_gold.jsonl \
    --mode hybrid \
    --output-dir output/rq4_causal

# Multi-hop 추론 평가
python -m benchmarking.experiments.multihop_eval \
    --gold-file data/multihop_gold.jsonl \
    --output-dir output/rq4_multihop
```

#### Baseline Comparisons

**PathRAG 벤치마크:**
```bash
python -m benchmarking.experiments.pathrag_lt_benchmark \
    --corpus path/to/corpus.jsonl \
    --qa-file path/to/qa_dataset.jsonl \
    --output-dir output/pathrag_results

# CROP 데이터셋 평가
python -m benchmarking.experiments.crop_pathrag_eval \
    --dataset crop \
    --output-dir output/crop_pathrag
```

**LightRAG 비교:**
```bash
python -m benchmarking.experiments.lightrag_comparison \
    --method proposed \
    --baseline lightrag \
    --output-dir output/lightrag_comparison

# CROP 데이터셋 평가
python -m benchmarking.experiments.crop_lightrag_eval \
    --dataset crop \
    --output-dir output/crop_lightrag
```

**Seed 전략 벤치마크:**
```bash
python -m benchmarking.experiments.seed_strategy_benchmark \
    --strategies all \
    --output-dir output/seed_results
```

#### Reference Only (Retrieval)

**BEIR 벤치마크 (RQ2 보완 - 외부 검색 벤치마크):**
```bash
# RQ2 검색 성능 보완: 외부 벤치마크로 일반화 검증
python -m benchmarking.experiments.beir_benchmark \
    --dataset smartfarm \
    --output-dir output/beir_results
```

#### Utilities

**배치 평가:**
```bash
python -m benchmarking.experiments.batch_eval_rag \
    --queries path/to/queries.jsonl \
    --endpoint http://localhost:8000/query \
    --output-dir output/batch_eval
```

**청킹 설정 평가:**
```bash
python -m benchmarking.experiments.eval_chunking_configs \
    --configs chunk_size_512,chunk_size_1024 \
    --corpus path/to/corpus.jsonl \
    --output-dir output/chunking_eval
```

**레이턴시 벤치마크:**
```bash
python -m benchmarking.experiments.ete_latency_benchmark \
    --iterations 100 \
    --output-dir output/latency_results
```

## Output Structure

실험 결과는 일반적으로 다음 구조로 저장됨:

```
output/
├── {experiment_name}/
│   ├── {timestamp}/
│   │   ├── results.json           # 상세 결과
│   │   ├── summary.md             # 요약 리포트
│   │   ├── metrics.json           # 메트릭 수치
│   │   └── experiment_config.json # 실험 설정
```

## Integration Points

- **데이터**: `benchmarking/data/smartfarm_eval.jsonl`
- **메트릭**: `benchmarking/metrics/`
- **리포터**: `benchmarking/reporters/`
- **설정**: `benchmarking/config/benchmark_config.yaml`

## Conventions

- 출력 디렉토리: `output/` (기본값)
- 결과 형식: JSON + Markdown
- 인코딩: UTF-8
- 타임스탬프: ISO 8601 형식 (`YYYY-MM-DD_HHMMSS`)
- Top-K 기본값: 4 (논문 섹션 5.1.3 기준)

## Notes

- 평가 데이터셋은 `benchmarking/data/` 디렉토리에 위치
- Docker 환경 설정은 `docker/` 디렉토리 참조
- 재현 가능성을 위해 random seed 설정 (config.runtime.seed = 42)
- 실험 결과는 Git에서 제외 (`.gitignore` 참조)
