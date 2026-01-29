#!/usr/bin/env bash
# 스마트팜 RAG Edge 프로파일링 러너 (4060Ti 8GB 등 8GB 환경용)
#
# - RunAllExperiments.sh 전체 플로우 실행 동안 GPU/CPU/메모리 사용량을 주기적으로 로그로 저장
# - 8GB 데스크톱 GPU 환경에서 실험 조합별 자원 프로파일링을 수행하고,
#   나중에 Jetson 등에서 핵심 조합만 재측정하는 용도로 사용
#
# 사용 예:
#   CHUNK_SIZE=5 CHUNK_STRIDE=2 \
#   scripts/EdgeProfilingRunner.sh \
#     --host http://127.0.0.1:41177 \
#     --input scripts/data/smartfarm_eval.jsonl \
#     --out-dir /tmp/era_rag_profile_cs5_st2 \
#     --label cs5_st2
#
# 로그 출력:
#   ${OUT_DIR}/profile/${LABEL}_nvidia_smi.csv   : nvidia-smi 기반 GPU/전력 프로파일 (가능한 경우)
#   ${OUT_DIR}/profile/${LABEL}_sysmon.log       : CPU/메모리 상위 프로세스 + free -m 스냅샷
#   ${OUT_DIR}/profile/run.log                   : RunAllExperiments.sh 전체 콘솔 로그

set -u

HOST=""
INPUT=""
OUT_DIR=""
LABEL="default"

usage() {
  echo "Usage: $0 --host HOST --input PATH --out-dir DIR [--label NAME]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2;;
    --input)
      INPUT="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --label)
      LABEL="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[WARN] Unknown argument: $1" >&2
      shift 1;;
  esac
done

if [[ -z "$HOST" || -z "$INPUT" || -z "$OUT_DIR" ]]; then
  echo "[ERROR] --host, --input, --out-dir는 필수입니다." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR/profile"

PROFILE_PREFIX="$OUT_DIR/profile/${LABEL}"
RUN_LOG="$OUT_DIR/profile/run.log"
GPU_LOG="${PROFILE_PREFIX}_nvidia_smi.csv"
SYS_LOG="${PROFILE_PREFIX}_sysmon.log"

echo "[INFO] Host=$HOST" | tee "$RUN_LOG"
echo "[INFO] Input=$INPUT" | tee -a "$RUN_LOG"
echo "[INFO] OutDir=$OUT_DIR" | tee -a "$RUN_LOG"
echo "[INFO] Label=$LABEL" | tee -a "$RUN_LOG"

GPU_MON_PID=""
SYS_MON_PID=""

# GPU 모니터 (nvidia-smi가 있을 때만)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] Starting nvidia-smi monitor..." | tee -a "$RUN_LOG"
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 1 \
    >"$GPU_LOG" 2>/dev/null &
  GPU_MON_PID=$!
else
  echo "[WARN] nvidia-smi not found; GPU metrics will not be recorded." | tee -a "$RUN_LOG"
fi

# 시스템(CPU/메모리) 모니터
{
  while true; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "--- ps top 10 by CPU ---"
    ps -eo pid,%cpu,%mem,command --sort=-%cpu | head -n 10 || true
    echo "--- free -m ---"
    free -m || true
    echo
    sleep 1
  done
} >"$SYS_LOG" 2>&1 &
SYS_MON_PID=$!

echo "[INFO] System monitor PID=$SYS_MON_PID" | tee -a "$RUN_LOG"
if [[ -n "$GPU_MON_PID" ]]; then
  echo "[INFO] GPU monitor PID=$GPU_MON_PID" | tee -a "$RUN_LOG"
fi

cleanup() {
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    if kill -0 "$GPU_MON_PID" 2>/dev/null; then
      kill "$GPU_MON_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "${SYS_MON_PID:-}" ]]; then
    if kill -0 "$SYS_MON_PID" 2>/dev/null; then
      kill "$SYS_MON_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

# 메인 실험 실행 (RunAllExperiments.sh)
echo "[INFO] Starting RunAllExperiments.sh..." | tee -a "$RUN_LOG"
"$SCRIPT_DIR/RunAllExperiments.sh" \
  --host "$HOST" \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" 2>&1 | tee -a "$RUN_LOG"

echo "[INFO] Experiments finished. Logs written under $OUT_DIR/profile" | tee -a "$RUN_LOG"
