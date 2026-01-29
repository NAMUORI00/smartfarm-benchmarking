#!/usr/bin/env bash
# Edge 환경에서 핵심 조합들을 한 번에 프로파일링하는 헬퍼 스크립트
# - 내부적으로 EdgeProfilingRunner.sh + RunAllExperiments.sh를 호출
# - CHUNK_SIZE/STRIDE 조합과 레이블을 사전 정의해 반복 실행한다.
#
# 사용 예:
#   bash scripts/run_edge_experiments.sh \
#     --host http://127.0.0.1:41177 \
#     --input scripts/data/smartfarm_eval.jsonl \
#     --base-out-dir /tmp/era_rag_edge_profiles

set -e

HOST="http://127.0.0.1:41177"
INPUT=""
BASE_OUT_DIR="/tmp/era_rag_edge_profiles"

usage() {
  echo "Usage: $0 --host HOST --input PATH [--base-out-dir DIR]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2;;
    --input)
      INPUT="$2"; shift 2;;
    --base-out-dir)
      BASE_OUT_DIR="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[WARN] Unknown argument: $1" >&2
      shift 1;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "[ERROR] --input JSONL 경로를 지정해야 합니다." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 조합 정의: label:CHUNK_SIZE:CHUNK_STRIDE
CONFIGS=(
  "cs5_st2:5:2"
  "cs6_st3:6:3"
)

mkdir -p "$BASE_OUT_DIR"

for conf in "${CONFIGS[@]}"; do
  IFS=":" read -r LABEL CS STRIDE <<<"$conf"
  OUT_DIR="$BASE_OUT_DIR/$LABEL"
  echo "[INFO] Running config $LABEL (CHUNK_SIZE=$CS, STRIDE=$STRIDE) -> $OUT_DIR"

  CHUNK_SIZE="$CS" CHUNK_STRIDE="$STRIDE" \
    "$SCRIPT_DIR/EdgeProfilingRunner.sh" \
      --host "$HOST" \
      --input "$INPUT" \
      --out-dir "$OUT_DIR" \
      --label "$LABEL"

done

echo "[INFO] All edge experiment runs finished. Base out dir: $BASE_OUT_DIR"
