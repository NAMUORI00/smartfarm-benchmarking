#!/usr/bin/env python3
"""
스마트팜 RAG 오프라인 배치 평가 스크립트 (D2 + F1 일부)

- /query 서버가 떠 있다는 전제에서, JSONL 형식의 질의 세트를 일괄 평가
- 논문용 지표에 바로 쓸 수 있는 최소 항목을 계산
  - 질의별: latency, answer/sources 텍스트, keyword hit 여부
  - 전체/카테고리별: 성공률, latency 통계, keyword hit 비율

JSONL 입력 형식 예시:

  {"id": "q1", "category": "온도", "question": "토마토 온실 생육 최적 온도는?",
   "expected_keywords": ["토마토", "온도", "도"]}

- expected_keywords 는 선택이며, 없으면 keyword 기반 hit 지표는 계산하지 않음.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmarking.utils.experiment_utils import (
    DEFAULT_BASE_URL,
    call_query,
    keyword_hits,
    load_queries,
    summarize,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_BASE_URL, help="/query 서버 base URL")
    ap.add_argument("--input", type=Path, default=None, help="평가용 질의 JSONL 경로")
    ap.add_argument("--ranker", default="none", help="사용할 리랭커 (none|llm|bge|llm-lite)")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("/tmp/batch_eval_results.json"))
    args = ap.parse_args()

    queries = load_queries(args.input)
    print(f"총 질의 수: {len(queries)} (입력 파일: {args.input or '내장 샘플'})")
    print(f"BASE_URL = {args.host}, ranker = {args.ranker}, top_k = {args.top_k}")

    all_results: List[Dict[str, Any]] = []
    for q in queries:
        qid = q.get("id") or q.get("question")[:16]
        category = q.get("category", "-")
        question = q["question"]
        expected = q.get("expected_keywords")
        print(f"\n[{qid}] ({category}) {question}")

        res = call_query(args.host, question, args.ranker, args.top_k)
        hits = keyword_hits(expected, res.get("answer", ""), res.get("sources", []))
        row: Dict[str, Any] = {
            "id": qid,
            "category": category,
            "question": question,
            "ranker": args.ranker,
            **res,
            **hits,
        }
        all_results.append(row)

        if res.get("success"):
            print(
                f"  ✓ {res['latency']:.3f}s | sources={res['num_sources']} | "
                f"hit_ans={hits['hit_in_answer']} hit_src={hits['hit_in_sources']}"
            )
        else:
            print(f"  ✗ ERROR: {res.get('error')}")
        time.sleep(0.3)

    # 전체 요약
    overall = summarize(all_results)
    print("\n=== 전체 요약 ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    # 카테고리별 요약
    cats = sorted({r.get("category", "-") for r in all_results})
    print("\n=== 카테고리별 요약 ===")
    per_cat: Dict[str, Any] = {}
    for c in cats:
        subset = [r for r in all_results if r.get("category", "-") == c]
        per_cat[c] = summarize(subset)
    print(json.dumps(per_cat, ensure_ascii=False, indent=2))

    # 결과 저장
    payload = {"overall": overall, "per_query": all_results, "per_category": per_cat}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
