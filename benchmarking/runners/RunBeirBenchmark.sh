#!/usr/bin/env bash
set -euo pipefail

RAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$RAG_DIR/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not found/executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to your venv python (e.g., PYTHON_BIN=python)." >&2
  exit 1
fi

DATASETS_DEFAULT=(scifact nfcorpus arguana fiqa)
if [[ -n "${DATASETS:-}" ]]; then
  IFS=' ' read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=("${DATASETS_DEFAULT[@]}")
fi

EMBED_MODEL_ID="${EMBED_MODEL_ID:-BAAI/bge-base-en-v1.5}"
SPARSE_METHOD="${SPARSE_METHOD:-bm25}"
BEIR_DIR="${BEIR_DIR:-$RAG_DIR/data/beir}"
OUTPUT_DIR="${OUTPUT_DIR:-$RAG_DIR/output/beir}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/run.log}"
MAX_QUERIES="${MAX_QUERIES:-}"
DOC_LIMIT="${DOC_LIMIT:-}"
RETRIEVAL_K="${RETRIEVAL_K:-100}"
DOWNLOAD="${DOWNLOAD:-0}"
USE_NOHUP="${USE_NOHUP:-0}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  env PYTHONPATH="${RAG_DIR}"
  "${PYTHON_BIN}" -m benchmarking.experiments.beir_benchmark
  --datasets "${DATASET_LIST[@]}"
  --beir-dir "${BEIR_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --embed-model "${EMBED_MODEL_ID}"
  --sparse-method "${SPARSE_METHOD}"
  --retrieval-k "${RETRIEVAL_K}"
)

if [[ -n "${MAX_QUERIES}" ]]; then
  CMD+=(--max-queries "${MAX_QUERIES}")
fi
if [[ -n "${DOC_LIMIT}" ]]; then
  CMD+=(--doc-limit "${DOC_LIMIT}")
fi
if [[ "${DOWNLOAD}" == "1" ]]; then
  CMD+=(--download)
fi

echo "[INFO] RAG_DIR=${RAG_DIR}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] Datasets: ${DATASET_LIST[*]}"
echo "[INFO] embed=${EMBED_MODEL_ID} sparse=${SPARSE_METHOD} retrieval_k=${RETRIEVAL_K}"

if [[ "${USE_NOHUP}" == "1" ]]; then
  nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
  echo "[OK] BEIR benchmark started (background). PID=$!"
  echo "[OK] Log: ${LOG_FILE}"
else
  "${CMD[@]}"
fi

