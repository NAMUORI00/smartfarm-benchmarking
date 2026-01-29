#!/usr/bin/env python3
"""PathRAG / Cache ablation 결과 요약 리포터

RunAllExperiments.sh 또는 EdgeProfilingRunner.sh를 다양한 설정으로 실행했을 때,
각 OUT_DIR 하위의 batch 결과를 읽어 ENABLE_PATHRAG, ENABLE_CACHE 조합별
성능(latency, success_rate)을 표로 요약한다.

사용 예:
  python scripts/AblationReporter.py \
    --base /tmp/era_rag_ablation_runs \
    --format markdown > ablation_summary.md

디렉터리 구조 예:
  /tmp/era_rag_ablation_runs/
    path_on_cache_on/
      batch/batch_none_top4.json
    path_on_cache_off/
      batch/batch_none_top4.json
    path_off_cache_on/
      batch/batch_none_top4.json
    path_off_cache_off/
      batch/batch_none_top4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_overall(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("overall", {})


def collect_rows(base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not base.exists():
        return rows

    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        lower = name.lower()
        path_on = "path_on" in lower or "pathrag_on" in lower
        cache_on = "cache_on" in lower

        batch_path = run_dir / "batch" / "batch_none_top4.json"
        overall = load_overall(batch_path)
        if not overall:
            continue

        rows.append(
            {
                "config": name,
                "enable_pathrag": path_on,
                "enable_cache": cache_on,
                "num_queries": overall.get("num_queries"),
                "success_rate": overall.get("success_rate"),
                "latency_mean": overall.get("latency_mean"),
                "latency_p95": overall.get("latency_p95"),
            }
        )
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    headers = [
        "config",
        "enable_pathrag",
        "enable_cache",
        "num_queries",
        "success_rate",
        "latency_mean",
        "latency_p95",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + " | ".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = [
            _fmt(r.get("config")),
            _fmt(r.get("enable_pathrag")),
            _fmt(r.get("enable_cache")),
            _fmt(r.get("num_queries")),
            _fmt(r.get("success_rate")),
            _fmt(r.get("latency_mean")),
            _fmt(r.get("latency_p95")),
        ]
        print("| " + " | ".join(vals) + " |")


def print_text(rows: List[Dict[str, Any]]) -> None:
    import json as _json

    for r in rows:
        print(_json.dumps(r, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize PathRAG/cache ablation runs.")
    ap.add_argument("--base", type=Path, required=True, help="실험 run 디렉터리들이 모인 상위 디렉터리")
    ap.add_argument("--format", choices=["markdown", "text"], default="markdown")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.base)
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_text(rows)


if __name__ == "__main__":
    main()
