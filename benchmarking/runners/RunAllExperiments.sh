#!/usr/bin/env bash
set -e

# ERA 스마트팜 RAG 주요 실험 올인원 실행 스크립트
# ------------------------------------------------
# 전제:
#   1) FastAPI 서버가 이미 실행 중이어야 함 (예: uvicorn app.main:app --host 0.0.0.0 --port 41177)
#   2) 평가용 JSONL 파일이 준비되어 있어야 함 (예: scripts/data/smartfarm_eval.jsonl)
#   3) python 환경(venv/conda 등)이 활성화되어 있어야 함
#
# 사용 예:
#   bash scripts/RunAllExperiments.sh \
#     --host http://127.0.0.1:41177 \
#     --input scripts/data/smartfarm_eval.jsonl \
#     --out-dir /tmp/era_rag_results
#
# 결과:
#   - 청킹 평가: /tmp/era_rag_results/chunking/chunking_eval_results.json
#   - 단일 설정 배치 평가: /tmp/era_rag_results/batch/*.json
#   - 다중 실험 결과: /tmp/era_rag_results/experiments/*.json

HOST="http://127.0.0.1:41177"
INPUT=""
OUT_DIR="/tmp/era_rag_results"
ABLATE_PATH_CACHE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2;;
    --input)
      INPUT="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --ablate-path-cache)
      ABLATE_PATH_CACHE="true"; shift 1;;
    *)
      echo "알 수 없는 인자: $1"; exit 1;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "[ERROR] --input JSONL 경로를 지정해야 합니다." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/chunking" "$OUT_DIR/batch" "$OUT_DIR/experiments" ""
run_single_profile() {
  local host="$1" input="$2" out_dir="$3"

  mkdir -p "$out_dir" "$out_dir/chunking" "$out_dir/batch" "$out_dir/experiments" ""

  echo "[INFO] Host=$host"
  echo "[INFO] Input JSONL=$input"
  echo "[INFO] Output dir=$out_dir"

  # 1. 청킹 조합 평가
  echo "\n[STEP 1] CHUNK_SIZE/STRIDE 조합 평가..."
  python scripts/eval_chunking_configs.py \
    --host "$host" \
    --configs 5,2 4,2 6,3 | tee "$out_dir/chunking/chunking_eval.log"

  if [[ -f /tmp/chunking_eval_results.json ]]; then
    mv /tmp/chunking_eval_results.json "$out_dir/chunking/chunking_eval_results.json"
  fi

  if [[ -f "$out_dir/chunking/chunking_eval_results.json" ]]; then
    echo "\n[STEP 1b] 청킹 튜닝 결과 요약(Markdown)..."
    python scripts/ChunkingExperimentReporter.py \
      --input "$out_dir/chunking/chunking_eval_results.json" \
      --format markdown > "$out_dir/chunking/chunking_summary.md"
  fi

  # 2. 단일 설정 배치 평가 (ranker=none/llm, top_k=4)
  echo "\n[STEP 2] 기본 RAG 성능 배치 평가..."
  python scripts/batch_eval_rag.py \
    --host "$host" \
    --input "$input" \
    --ranker none \
    --top_k 4 \
    --output "$out_dir/batch/batch_none_top4.json"

  python scripts/batch_eval_rag.py \
    --host "$host" \
    --input "$input" \
    --ranker llm \
    --top_k 4 \
    --output "$out_dir/batch/batch_llm_top4.json"

  # 3. 다중 실험 러너 (none/llm, top_k=4)
  echo "\n[STEP 3] RagExperimentRunner로 다중 실험 실행..."
  python scripts/RagExperimentRunner.py \
    --host "$host" \
    --input "$input" \
    --rankers none,llm \
    --top-k 4 \
    --output-dir "$out_dir/experiments"

}


echo "[INFO] Host=$HOST"
echo "[INFO] Input JSONL=$INPUT"
echo "[INFO] Output base dir=$OUT_DIR"

if [[ "$ABLATE_PATH_CACHE" != "true" ]]; then
  echo "\n[MODE] 단일 프로파일 실행 (PathRAG/cache 현재 env 그대로)"
  mkdir -p "$OUT_DIR" "$OUT_DIR/chunking" "$OUT_DIR/batch" "$OUT_DIR/experiments"
  run_single_profile "$HOST" "$INPUT" "$OUT_DIR"
  echo "\n[OK] 모든 실험이 완료되었습니다. 결과 폴더: $OUT_DIR"
  exit 0
fi

# Ablation 모드: ENABLE_PATHRAG / ENABLE_CACHE 조합별로 반복 실행

echo "\n[MODE] Ablation 모드: PathRAG × Cache 조합 실행"

for path_flag in on off; do
  for cache_flag in on off; do
    if [[ "$path_flag" == "on" ]]; then
      export ENABLE_PATHRAG=true
    else
      export ENABLE_PATHRAG=false
    fi
    if [[ "$cache_flag" == "on" ]]; then
      export ENABLE_CACHE=true
    else
      export ENABLE_CACHE=false
    fi

    run_name="path_${path_flag}_cache_${cache_flag}"
    run_dir="$OUT_DIR/${run_name}"

    echo "\n----------------------------------------"
    echo "[RUN] $run_name (ENABLE_PATHRAG=$ENABLE_PATHRAG, ENABLE_CACHE=$ENABLE_CACHE)"
    echo "[RUN] Output dir=$run_dir"
    mkdir -p "$run_dir"

    run_single_profile "$HOST" "$INPUT" "$run_dir"
  done
done

echo "\n[OK] Ablation 모드 모든 실험 완료. 베이스 폴더: $OUT_DIR"
