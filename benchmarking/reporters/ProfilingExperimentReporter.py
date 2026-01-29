#!/usr/bin/env python3
"""Edge/RAG 실험 자원 프로파일링 리포터

- EdgeProfilingRunner.sh + RunAllExperiments.sh 실행 결과 디렉터리들을 읽어
  논문용으로 바로 사용할 수 있는 요약 Markdown 표를 생성한다.
- 입력 디렉터리 구조(각 DIR 마다):
  DIR/
    batch/
      batch_*.json           # batch_eval_rag.py 결과 (overall/per_query/per_category)
    experiments/
      *.json                 # RagExperimentRunner.py 결과 (옵션)
    profile/
      <label>_nvidia_smi.csv # GPU/전력 로그 (옵션)
      <label>_sysmon.log     # CPU/메모리 로그 (옵션)

사용 예:
  python scripts/ProfilingExperimentReporter.py \
    --dirs /tmp/era_rag_profile_cs5_st2 /tmp/era_rag_profile_cs6_st3 \
    --format markdown > profiling_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_float(val: str | None) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # "12.3 W" 또는 "45 %" 형태를 처리
    first = s.split()[0]
    try:
        return float(first)
    except ValueError:
        return None


def parse_nvidia_smi_csv(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    gpu_util: List[float] = []
    mem_used: List[float] = []
    power_draw: List[float] = []

    for row in rows:
        for key in list(row.keys()):
            v = row[key]
            # DictReader가 BOM 등을 포함한 컬럼명을 가질 수 있어 키 정규화는 값 파싱에서 처리
        u = _safe_float(row.get("utilization.gpu [%]"))
        mu = _safe_float(row.get("memory.used [MiB]"))
        pw = _safe_float(row.get("power.draw [W]"))
        if u is not None:
            gpu_util.append(u)
        if mu is not None:
            mem_used.append(mu)
        if pw is not None:
            power_draw.append(pw)

    out: Dict[str, float] = {}
    if gpu_util:
        out["gpu_util_mean"] = statistics.mean(gpu_util)
        out["gpu_util_max"] = max(gpu_util)
    if mem_used:
        out["gpu_mem_used_mean"] = statistics.mean(mem_used)
        out["gpu_mem_used_max"] = max(mem_used)
    if power_draw:
        out["power_mean"] = statistics.mean(power_draw)
        out["power_max"] = max(power_draw)
    return out


_MEM_RE = re.compile(r"^Mem:\s+(\d+)\s+(\d+)\s+(\d+)")


def parse_sysmon_log(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    used: List[float] = []
    total: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _MEM_RE.match(line.strip())
            if not m:
                continue
            tot, use, _free = m.groups()
            try:
                total.append(float(tot))
                used.append(float(use))
            except ValueError:
                continue
    out: Dict[str, float] = {}
    if used:
        out["sys_mem_used_mean"] = statistics.mean(used)
        out["sys_mem_used_max"] = max(used)
    if total:
        out["sys_mem_total_mean"] = statistics.mean(total)
    return out


def load_batch_summaries(batch_dir: Path) -> List[Dict[str, Any]]:
    if not batch_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        overall = data.get("overall", {})
        per_query = data.get("per_query") or []
        # ranker/top_k는 per_query 또는 파일명에서 추출
        ranker = None
        top_k = None
        if per_query:
            ranker = per_query[0].get("ranker")
        if ranker is None:
            # 파일 이름 예: batch_none_top4.json
            stem = path.stem
            parts = stem.split("_")
            if len(parts) >= 3:
                ranker = parts[1]
                if parts[2].startswith("top"):
                    try:
                        top_k = int(parts[2][3:])
                    except ValueError:
                        top_k = None
        row = {
            "file": str(path),
            "ranker": ranker or "-",
            "top_k": top_k,
            "overall": overall,
        }
        out.append(row)
    return out


def collect_rows(dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in dirs:
        batch_rows = load_batch_summaries(d / "batch")
        profile_dir = d / "profile"
        gpu_csv = next(iter(profile_dir.glob("*_nvidia_smi.csv")), None) if profile_dir.exists() else None
        sys_log = next(iter(profile_dir.glob("*_sysmon.log")), None) if profile_dir.exists() else None

        gpu_stats = parse_nvidia_smi_csv(gpu_csv) if gpu_csv else {}
        sys_stats = parse_sysmon_log(sys_log) if sys_log else {}

        for b in batch_rows:
            overall = b["overall"] or {}
            row: Dict[str, Any] = {
                "config": d.name,
                "ranker": b["ranker"],
                "top_k": b["top_k"],
                "num_queries": overall.get("num_queries"),
                "success_rate": overall.get("success_rate"),
                "latency_mean": overall.get("latency_mean"),
                "latency_p95": overall.get("latency_p95"),
            }
            row.update(gpu_stats)
            row.update(sys_stats)
            rows.append(row)
    return rows


def _fmt(val: Any, digits: int = 3) -> str:
    if val is None:
        return "-";
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    headers = [
        "config",
        "ranker",
        "top_k",
        "num_queries",
        "success_rate",
        "latency_mean",
        "latency_p95",
        "gpu_util_mean",
        "gpu_mem_used_mean",
        "power_mean",
        "sys_mem_used_max",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + " | ".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = [
            _fmt(r.get("config"), 0),
            _fmt(r.get("ranker"), 0),
            _fmt(r.get("top_k"), 0),
            _fmt(r.get("num_queries"), 0),
            _fmt(r.get("success_rate")),
            _fmt(r.get("latency_mean")),
            _fmt(r.get("latency_p95")),
            _fmt(r.get("gpu_util_mean")),
            _fmt(r.get("gpu_mem_used_mean")),
            _fmt(r.get("power_mean")),
            _fmt(r.get("sys_mem_used_max")),
        ]
        print("| " + " | ".join(vals) + " |")


def print_text(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("no rows")
        return
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize edge RAG profiling runs into a table.")
    ap.add_argument("--dirs", nargs="+", type=Path, help="EdgeProfilingRunner OUT_DIR 목록")
    ap.add_argument("--format", choices=["markdown", "text"], default="markdown")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.dirs)
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_text(rows)


if __name__ == "__main__":
    main()
