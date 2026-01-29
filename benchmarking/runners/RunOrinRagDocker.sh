#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

KEEP_STACK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-stack)
      KEEP_STACK=true
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 [--keep-stack]" >&2
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export DNS1="${DNS1:-1.1.1.1}"
export DNS2="${DNS2:-8.8.8.8}"
export DEVICE="${DEVICE:-cuda}"
export EMBED_MODEL_ID="${EMBED_MODEL_ID:-minilm}"
export EMBED_CACHE_SIZE="${EMBED_CACHE_SIZE:-256}"
export CTX_SIZE="${CTX_SIZE:-4096}"
export GPU_LAYERS="${GPU_LAYERS:--1}"
export LLM_GGUF="${LLM_GGUF:-Qwen3-4B-Q4_K_M.gguf}"
export LLM_PARALLEL="${LLM_PARALLEL:-2}"

cd "$ROOT_DIR"

echo "[INFO] Starting Orin stack for RAG experiments..."
docker compose -f docker-compose.yml -f docker-compose.orin.yml up -d api llama

echo "[INFO] Running RAG experiments (runner profile)..."
docker compose -f docker-compose.yml -f docker-compose.orin.yml --profile runner run --rm rag-runner

if ! $KEEP_STACK; then
  echo "[INFO] Stopping stack..."
  docker compose -f docker-compose.yml -f docker-compose.orin.yml down
fi
