#!/usr/bin/env python3
"""청킹 튜닝 실험 결과 요약 스크립트

- 입력: eval_chunking_configs.py 또는 RunAllExperiments.sh가 생성한 JSON 결과
  (기본: /tmp/era_rag_results/chunking/chunking_eval_results.json)
- 출력: 텍스트 또는 Markdown 표 형식 요약

논문 작성 시 "CHUNK_SIZE/STRIDE vs latency" 표를 바로 만들 수 있도록 설계.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_results(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"청킹 결과 JSON을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("결과 JSON 형식이 dict가 아닙니다.")
    return data


def summarize_results(data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cfg_key, records in data.items():
        latencies = [r["latency"] for r in records if "latency" in r]
        sources = [r.get("sources") for r in records if "sources" in r]
        if not latencies:
            continue
        latencies_sorted = sorted(latencies)
        mean = statistics.mean(latencies)
        median = statistics.median(latencies)
        # 간단 p95 추정 (샘플 수가 적어도 동작하도록 방어적 처리)
        idx_95 = max(0, int(round(0.95 * (len(latencies_sorted) - 1))))
        p95 = latencies_sorted[idx_95]
        src_mean = sum(s for s in sources if isinstance(s, (int, float))) / len(sources) if sources else None

        # cfg_key: "cs=5,st=2" 형식 가정
        cs, st = None, None
        try:
            parts = cfg_key.split(",")
            cs = int(parts[0].split("=")[1])
            st = int(parts[1].split("=")[1])
        except Exception:
            pass

        rows.append(
            {
                "key": cfg_key,
                "chunk_size": cs,
                "chunk_stride": st,
                "n": len(latencies),
                "lat_mean": mean,
                "lat_median": median,
                "lat_p95": p95,
                "src_mean": src_mean,
            }
        )

    # 평균 latency 기준 오름차순 정렬
    rows.sort(key=lambda r: r["lat_mean"])
    return rows


def print_text(rows: List[Dict[str, Any]]) -> None:
    print("청킹 튜닝 결과 요약 (latency 기준 오름차순)")
    print("=" * 72)
    for r in rows:
        cs = r["chunk_size"]
        st = r["chunk_stride"]
        print(
            f"CHUNK_SIZE={cs}, STRIDE={st} | "
            f"n={r['n']} | mean={r['lat_mean']:.3f}s | median={r['lat_median']:.3f}s | p95={r['lat_p95']:.3f}s | "
            f"avg_sources={r['src_mean']:.2f}"
        )


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    print("| CHUNK_SIZE | CHUNK_STRIDE | n | mean_latency(s) | median_latency(s) | p95_latency(s) | avg_sources |")
    print("|-----------:|-------------:|--:|---------------:|------------------:|---------------:|-----------:|")
    for r in rows:
        cs = r["chunk_size"]
        st = r["chunk_stride"]
        print(
            f"| {cs} | {st} | {r['n']} | "
            f"{r['lat_mean']:.3f} | {r['lat_median']:.3f} | {r['lat_p95']:.3f} | {r['src_mean']:.2f} |"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="청킹 튜닝 결과 요약 리포터")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/era_rag_results/chunking/chunking_eval_results.json"),
        help="eval_chunking_configs.py가 생성한 JSON 경로",
    )
    ap.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="출력 형식 (text | markdown)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data = load_results(args.input)
    rows = summarize_results(data)
    if not rows:
        print("요약할 데이터가 없습니다.")
        return
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_text(rows)


if __name__ == "__main__":
    main()
