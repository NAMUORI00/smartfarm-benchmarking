
## Script categorization (server / llm / experiments / tools)

- Server utilities (RAG API / stack):
  - `run_server_dev.sh`
  - `run_server_edge.sh`
  - `run_server_jetson.sh`
  - `run_edge_stack.sh`

- LLM utilities (local llama.cpp):
  - `run-llama-local.sh`
  - `llama-local-manage.sh`

- Experiments / profiling:
  - `RunAllExperiments.sh`
  - `RunBeirBenchmark.sh`
  - `EdgeProfilingRunner.sh`
  - `run_edge_experiments.sh`
  - `batch_eval_rag.py`
  - `RagExperimentRunner.py`
  - `eval_chunking_configs.py`
  - `ChunkingExperimentReporter.py`
  - `AblationReporter.py`
  - `ProfilingExperimentReporter.py`

- Tools / data utilities:
  - `ingest_smartfarm_kb.py`
  - `experiment_utils.py`

# ERA 스마트팜 RAG 실험 스크립트 모음

이 디렉터리에는 ERA(Edge-RAG-Anything) 3.1/3.2 기준 스마트팜 RAG 실험을 위한 스크립트들이 들어 있습니다.

## 공통 전제

- FastAPI 서버(`/query`)가 먼저 떠 있어야 합니다.
- 기본 호스트: `http://127.0.0.1:41177` (필요 시 `--host`로 변경)
- 평가용 질의 세트: JSONL
  - 필드 예시: `id`, `category`, `question`, `expected_keywords`

---

## 1. 청킹 조합 평가: `eval_chunking_configs.py` + `ChunkingExperimentReporter.py`

- 목적: `CHUNK_SIZE`, `CHUNK_STRIDE` 조합에 따른 응답 시간·소스 개수 비교 및 논문용 표 생성.
- 방식:
  - 서버가 환경변수 `CHUNK_SIZE`, `CHUNK_STRIDE`를 읽는다는 전제에서,
  - 대표 질의 집합(`QUERIES`)을 고정하고 조합별로 `/query`를 호출 (`eval_chunking_configs.py`).
  - 생성된 JSON 결과를 `ChunkingExperimentReporter.py`로 요약(text/markdown 표).
- 출력:
  - `eval_chunking_configs.py`:
    - 터미널: 각 조합별 평균 latency, 질의별 latency/sources.
    - 파일: `/tmp/chunking_eval_results.json` 또는 `RunAllExperiments.sh` 경로 기준 `${OUT_DIR}/chunking/chunking_eval_results.json`.
  - `ChunkingExperimentReporter.py`:
    - 터미널: 텍스트 또는 Markdown 표 형식 요약.
- 사용 예:
  - 평가 실행: `python scripts/eval_chunking_configs.py --host http://127.0.0.1:41177 --configs 5,2 4,2 6,3`.
  - 요약 출력(text): `python scripts/ChunkingExperimentReporter.py --input /tmp/chunking_eval_results.json`.
  - 요약 출력(MD): `python scripts/ChunkingExperimentReporter.py --input /tmp/chunking_eval_results.json --format markdown`.

논문 활용 포인트:
- CHUNK_SIZE/STRIDE에 따른 latency–정확도(근거 개수) trade-off를 그림/표로 제시.
- Markdown 모드 출력은 논문에 들어갈 표의 초안으로 바로 사용 가능.

---

## 2. 단일 설정 배치 평가: `batch_eval_rag.py`

- 목적: 하나의 설정(rank·top_k)에 대해 JSONL 질의 세트를 일괄 평가.
- 역할:
  - `/query`에 대해 질의별 latency, answer, sources를 수집.
  - `expected_keywords`가 주어지면 keyword 기반 hit 여부(`hit_in_answer`, `hit_in_sources`)를 계산.
- 출력:
  - 터미널: 전체 요약(성공률, latency 통계) + 카테고리별 요약.
  - 파일: 기본 `/tmp/batch_eval_results.json`.
- 사용 예:
  - `python scripts/batch_eval_rag.py --host http://127.0.0.1:41177 --input scripts/data/smartfarm_eval.jsonl --ranker none --top_k 4 --output /tmp/batch_eval_none_top4.json`.

논문 활용 포인트:
- ranker/검색 파라미터 고정 시, 스마트팜 실사용 질의에 대한 기본 성능 및 latency 통계.

---

## 3. 다중 실험 러너 (PascalCase): `RagExperimentRunner.py`

- 목적: 여러 실험 설정을 한 번에 실행해 각각의 JSON 결과를 생성.
- 내부:
  - `ExperimentConfig`: `(name, ranker, top_k)` 보관.
  - `RagExperimentRunner`: 호스트/입력/출력 디렉터리를 받아 여러 실험 실행.
  - `batch_eval_rag` 모듈의 공통 함수(load_queries, call_query, keyword_hits, summarize)를 재사용.
- 출력:
  - `--output-dir` 안에 각 실험 이름(PascalCase 기반)으로 JSON 파일 생성.
  - 예: `NoneTop4.json`, `LlmTop4.json`.
- 사용 예:
  - `python scripts/RagExperimentRunner.py --host http://127.0.0.1:41177 --input smartfarm_eval.jsonl --rankers none,llm --top-k 4 --output-dir /tmp/rag_experiments`.

논문 활용 포인트:
- ranker/검색 파라미터 조합에 대한 비교 표·그래프 작성.

---

## 4. 인게스트 스크립트: `ingest_smartfarm_kb.py`

- 목적: 스마트팜 도메인 KB를 로컬 인덱스로 인게스트.
- 역할:
  - 도메인 문서들을 `SourceDoc` 형태로 변환 후 Sparse/Dense 인덱스에 추가.
- 사용 예:
  - `python scripts/ingest_smartfarm_kb.py --input data/smartfarm_docs` (실제 인자 구성은 스크립트 구현 참조).

논문 활용 포인트:
- 실험에 사용한 데이터셋/KB 구축 과정 설명.

---

## 5. 로컬 LLM 관리 스크립트: `run-llama-local.sh`, `llama-local-manage.sh`

- 목적: 로컬 llama.cpp 기반 LLM 서버 관리(시작/중지 등).
- 논문에는 직접적으로 사용되지 않지만, 재현 환경 구성 시 참고.

---

## 6. 올인원 실험 실행: `RunAllExperiments.sh`

- 목적: 서버가 떠 있는 상태에서 논문에 필요한 주요 실험을 한 번에 실행.
- 수행 순서(예시):
  1. 청킹 조합 평가(`eval_chunking_configs.py`).
  2. 기본 RAG 성능 평가(`batch_eval_rag.py` — ranker=none/llm).
  3. 다중 실험(`RagExperimentRunner.py` — none/llm, top_k=4).
- 상세 사용 방법은 `RunAllExperiments.sh` 상단 주석 참고.
