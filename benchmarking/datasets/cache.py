from __future__ import annotations

import os
from pathlib import Path


HF_CACHE_DIR_ENV = "SMARTFARM_BENCHMARK_HF_CACHE_DIR"


def resolve_hf_cache_dir(cache_dir: Path | None = None) -> Path:
    """Resolve HuggingFace datasets cache directory.

    Priority:
      1) `cache_dir` argument
      2) `SMARTFARM_BENCHMARK_HF_CACHE_DIR` env var
      3) <repo>/.cache/hf_datasets
    """
    if cache_dir is not None:
        resolved = Path(cache_dir).expanduser().resolve()
    else:
        from_env = os.getenv(HF_CACHE_DIR_ENV, "").strip()
        if from_env:
            resolved = Path(from_env).expanduser().resolve()
        else:
            # .../smartfarm-benchmarking/benchmarking/datasets/cache.py
            #                       ^ repo root is parents[2]
            resolved = Path(__file__).resolve().parents[2] / ".cache" / "hf_datasets"

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

