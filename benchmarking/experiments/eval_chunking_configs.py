#!/usr/bin/env python3
"""간단 CHUNK_SIZE/STRIDE 조합 평가 스크립트
- 서버( /query )가 떠 있다는 전제 하에, 대표 질의 세트를 돌려
  조합별 평균 응답시간과 sources 개수만 빠르게 비교
- 정량 평가(p@k 등)는 이후 D2 단계에서 확장
"""
import os
import time
import json
import argparse

import requests

QUERIES = [
    "토마토 온실 생육 최적 온도는?",
    "파프리카 양액 EC 기준을 알려줘",
    "딸기 흰가루병 초기 증상과 대처법은?",
    "상추 파종부터 수확까지 재배 일정을 정리해줘",
]


def run_eval(base_url: str, chunk_size: int, chunk_stride: int, top_k: int = 2):
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_STRIDE"] = str(chunk_stride)

    rows = []
    for q in QUERIES:
        payload = {"question": q, "top_k": top_k, "ranker": "none"}
        t0 = time.time()
        resp = requests.post(f"{base_url}/query", json=payload, timeout=60)
        dt = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            rows.append(
                {
                    "question": q,
                    "latency": round(dt, 3),
                    "sources": len(data.get("sources", [])),
                }
            )
        else:
            rows.append(
                {
                    "question": q,
                    "latency": round(dt, 3),
                    "error": f"HTTP {resp.status_code}",
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:41177")
    ap.add_argument("--configs", nargs="*", default=["5,2", "4,2", "6,3"])
    args = ap.parse_args()

    all_results = {}
    for cfg in args.configs:
        cs, st = map(int, cfg.split(","))
        key = f"cs={cs},st={st}"
        print(f"\n[CONFIG] {key}")
        rows = run_eval(args.host, cs, st)
        all_results[key] = rows
        lat_ok = [r["latency"] for r in rows if "latency" in r]
        if lat_ok:
            avg = sum(lat_ok) / len(lat_ok)
            print(f"  평균 응답시간: {avg:.3f}s")
        for r in rows:
            print(f"  - Q={r['question'][:20]}... | t={r['latency']:.3f}s | sources={r.get('sources','-')}")

    out_path = "/tmp/chunking_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
