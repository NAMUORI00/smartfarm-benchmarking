#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_DIR="$ROOT_DIR/benchmarking/docker"

ONLY="baseline,ablation,domain,edge"
DEVICES="gpu,cpu"
MODELS="minilm,Qwen/Qwen3-Embedding-0.6B"
OUTPUT_BASE="/app/output/orin_bench"

usage() {
  echo "Usage: $0 [--only EXP] [--devices gpu|cpu|both] [--models CSV] [--output-base PATH]" >&2
  echo "  --only        Comma-separated experiments (default: $ONLY)" >&2
  echo "  --devices     gpu|cpu|both (default: both)" >&2
  echo "  --models      Comma-separated EMBED_MODEL_ID list (default: $MODELS)" >&2
  echo "  --output-base Output base path inside container (default: $OUTPUT_BASE)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="$2"; shift 2;;
    --devices)
      case "$2" in
        gpu) DEVICES="gpu";;
        cpu) DEVICES="cpu";;
        both) DEVICES="gpu,cpu";;
        *) echo "[ERROR] Invalid --devices: $2" >&2; exit 1;;
      esac
      shift 2;;
    --models)
      MODELS="$2"; shift 2;;
    --output-base)
      OUTPUT_BASE="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

export DNS1="${DNS1:-1.1.1.1}"
export DNS2="${DNS2:-8.8.8.8}"
export EMBED_CACHE_SIZE="${EMBED_CACHE_SIZE:-256}"

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"

for device in "${DEVICE_LIST[@]}"; do
  case "$device" in
    gpu) service="benchmark-gpu"; device_env="cuda";;
    cpu) service="benchmark-cpu"; device_env="cpu";;
    *) echo "[ERROR] Unknown device: $device" >&2; exit 1;;
  esac

  for model in "${MODEL_LIST[@]}"; do
    model_tag="$(echo "$model" | tr '/:' '_' | tr '[:upper:]' '[:lower:]')"
    out_dir="${OUTPUT_BASE}/${device}_${model_tag}"
    echo "[INFO] Running $service (DEVICE=$device_env, EMBED_MODEL_ID=$model) -> $out_dir"

    DNS1="$DNS1" DNS2="$DNS2" EMBED_CACHE_SIZE="$EMBED_CACHE_SIZE" \
      docker compose -f "$DOCKER_DIR/docker-compose.yml" run --rm \
      -e DEVICE="$device_env" \
      -e EMBED_MODEL_ID="$model" \
      -e EMBED_CACHE_SIZE="$EMBED_CACHE_SIZE" \
      -e PYTHONUNBUFFERED=1 \
      "$service" \
      --config /app/benchmarking/config/benchmark_config.yaml \
      --only "$ONLY" \
      --output "$out_dir"
  done
done
