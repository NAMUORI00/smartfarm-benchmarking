#!/usr/bin/env python3
"""RAG Experiment Runner

여러 실험 설정을 한 번에 돌려서 논문용 결과(JSON)들을 일괄 생성하는 스크립트.

- 개별 실험은 scripts.batch_eval_rag 모듈의 함수(load_queries, call_query, keyword_hits, summarize)를 재사용
- 실험 단위: (ranker, top_k) 조합
- 출력: 실험별로 1개 JSON 파일 (overall / per_query / per_category)

사용 예시:

  python scripts/RagExperimentRunner.py \
    --host http://127.0.0.1:41177 \
    --input /path/to/smartfarm_eval.jsonl \
    --rankers none,llm \
    --top-k 4 \
    --output-dir /tmp/rag_experiments

"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from benchmarking.utils.experiment_utils import (
    DEFAULT_BASE_URL,
    call_query,
    keyword_hits,
    load_queries,
    summarize,
)


@dataclass
class ExperimentConfig:
    name: str
    ranker: str
    top_k: int


class RagExperimentRunner:
    def __init__(self, host: str, input_path: Path | None, output_dir: Path):
        self.host = host
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_single(self, cfg: ExperimentConfig) -> Dict[str, Any]:
        queries = load_queries(self.input_path)
        all_results: List[Dict[str, Any]] = []

        print(f"\n=== Experiment: {cfg.name} (ranker={cfg.ranker}, top_k={cfg.top_k}) ===")
        print(f"총 질의 수: {len(queries)}")

        for q in queries:
            qid = q.get("id") or q.get("question", "")[:16]
            category = q.get("category", "-")
            question = q["question"]
            expected = q.get("expected_keywords")
            print(f"[{qid}] ({category}) {question}")

            res = call_query(self.host, question, cfg.ranker, cfg.top_k)
            hits = keyword_hits(expected, res.get("answer", ""), res.get("sources", []))
            row: Dict[str, Any] = {
                "id": qid,
                "category": category,
                "question": question,
                "ranker": cfg.ranker,
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
            time.sleep(0.2)

        overall = summarize(all_results)
        cats = sorted({r.get("category", "-") for r in all_results})
        per_cat: Dict[str, Any] = {}
        for c in cats:
            subset = [r for r in all_results if r.get("category", "-") == c]
            per_cat[c] = summarize(subset)

        payload = {
            "config": {
                "name": cfg.name,
                "ranker": cfg.ranker,
                "top_k": cfg.top_k,
                "host": self.host,
                "input": str(self.input_path) if self.input_path is not None else None,
            },
            "overall": overall,
            "per_query": all_results,
            "per_category": per_cat,
        }

        out_path = self.output_dir / f"{cfg.name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"실험 결과 저장: {out_path}")

        return payload

    def run_all(self, configs: List[ExperimentConfig]) -> None:
        for cfg in configs:
            self._run_single(cfg)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run multiple RAG experiments and save JSON results.")
    ap.add_argument("--host", default=DEFAULT_BASE_URL, help="/query 서버 base URL")
    ap.add_argument("--input", type=Path, default=None, help="평가용 질의 JSONL 경로")
    ap.add_argument("--rankers", default="none,llm", help="콤마로 구분한 ranker 목록 (예: none,llm)")
    ap.add_argument("--top-k", type=int, default=4, help="검색 top_k")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/rag_experiments"))
    return ap.parse_args()


def build_default_configs(ranker_str: str, top_k: int) -> List[ExperimentConfig]:
    rankers = [r.strip() for r in ranker_str.split(",") if r.strip()]
    configs: List[ExperimentConfig] = []
    for r in rankers:
        # PascalCase 스타일 이름: 예) NoneTop4, LlmTop4
        name = f"{r.capitalize()}Top{top_k}"
        configs.append(ExperimentConfig(name=name, ranker=r, top_k=top_k))
    return configs


def main() -> None:
    args = parse_args()
    configs = build_default_configs(args.rankers, args.top_k)
    runner = RagExperimentRunner(args.host, args.input, args.output_dir)
    runner.run_all(configs)


if __name__ == "__main__":
    main()
