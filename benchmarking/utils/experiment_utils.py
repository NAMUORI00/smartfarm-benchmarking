#!/usr/bin/env python3
"""공통 RAG 실험 유틸리티.

- JSONL 질의 로딩
- /query 호출 및 latency 측정
- keyword 기반 hit 계산
- 전체/카테고리별 요약 통계

배치 평가(batch_eval_rag.py)와 다중 실험 러너(RagExperimentRunner.py)에서 재사용한다.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:41177"

# 기본 샘플 질의(입력 파일이 없을 때 사용)
FALLBACK_QUERIES: List[Dict[str, Any]] = [
    {"id": "q1", "category": "온도", "question": "토마토 온실 생육 최적 온도는?"},
    {"id": "q2", "category": "양액", "question": "파프리카 양액 EC 기준을 알려줘"},
    {"id": "q3", "category": "병해충", "question": "딸기 흰가루병 초기 증상과 대처법은?"},
    {"id": "q4", "category": "재배일정", "question": "상추 파종부터 수확까지 재배 일정을 정리해줘"},
]


def load_queries(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return FALLBACK_QUERIES
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def call_query(base_url: str, question: str, ranker: str, top_k: int) -> Dict[str, Any]:
    payload = {"question": question, "top_k": top_k, "ranker": ranker}
    t0 = time.time()
    resp = requests.post(f"{base_url}/query", json=payload, timeout=60)
    dt = time.time() - t0
    result: Dict[str, Any] = {"latency": dt, "status_code": resp.status_code}
    if resp.status_code == 200:
        data = resp.json()
        result["answer"] = data.get("answer", "")
        result["sources"] = data.get("sources", [])
        result["num_sources"] = len(result["sources"])
        result["success"] = True
    else:
        result["answer"] = ""
        result["sources"] = []
        result["num_sources"] = 0
        result["success"] = False
        result["error"] = f"HTTP {resp.status_code}"
    return result


def keyword_hits(expected: List[str] | None, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not expected:
        return {
            "has_expected": False,
            "hit_in_answer": False,
            "hit_in_sources": False,
            "num_hit_keywords_answer": 0,
            "num_hit_keywords_sources": 0,
        }
    answer_text = answer
    source_text = "\n".join(str(s.get("text", "")) for s in sources)
    hit_ans = 0
    hit_src = 0
    for kw in expected:
        if kw in answer_text:
            hit_ans += 1
        if kw in source_text:
            hit_src += 1
    return {
        "has_expected": True,
        "hit_in_answer": hit_ans > 0,
        "hit_in_sources": hit_src > 0,
        "num_hit_keywords_answer": hit_ans,
        "num_hit_keywords_sources": hit_src,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    latencies = [r["latency"] for r in results if r.get("success")]
    success = [r for r in results if r.get("success")]
    summary: Dict[str, Any] = {
        "num_queries": len(results),
        "num_success": len(success),
        "success_rate": (len(success) / len(results)) if results else 0.0,
    }
    if latencies:
        lat_sorted = sorted(latencies)
        summary["latency_mean"] = statistics.mean(latencies)
        summary["latency_min"] = lat_sorted[0]
        summary["latency_max"] = lat_sorted[-1]
        summary["latency_p50"] = lat_sorted[int(0.5 * (len(lat_sorted) - 1))]
        summary["latency_p95"] = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    with_kw = [r for r in results if r.get("has_expected")]
    if with_kw:
        summary["keyword_hit_answer_rate"] = sum(1 for r in with_kw if r.get("hit_in_answer")) / len(with_kw)
        summary["keyword_hit_sources_rate"] = sum(1 for r in with_kw if r.get("hit_in_sources")) / len(with_kw)
    return summary
