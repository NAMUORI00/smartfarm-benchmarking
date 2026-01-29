#!/usr/bin/env python3
"""RAGAS evaluation for the RAG system.

Calls the /query endpoint for each question, builds a RAGAS dataset, and
computes LLM-based metrics (faithfulness, answer relevancy, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from benchmarking.utils.experiment_utils import DEFAULT_BASE_URL, call_query
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

try:  # Optional dependency
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    HAS_RAGAS = True
except Exception:  # pragma: no cover - optional dependency at runtime
    HAS_RAGAS = False


DEFAULT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

_METRIC_MAP = {
    "faithfulness": faithfulness if HAS_RAGAS else None,
    "answer_relevancy": answer_relevancy if HAS_RAGAS else None,
    "context_precision": context_precision if HAS_RAGAS else None,
    "context_recall": context_recall if HAS_RAGAS else None,
    "answer_correctness": answer_correctness if HAS_RAGAS else None,
}


@dataclass
class QAItem:
    id: str
    question: str
    answer: str
    source_ids: List[str]


def _ensure_ragas_available() -> None:
    if HAS_RAGAS:
        return
    raise ImportError(
        "ragas가 설치되어 있지 않습니다. "
        "pip로 ragas, datasets, langchain-openai를 설치하세요 "
        "(예: `pip install ragas datasets langchain-openai`)."
    )


def _normalize_metric_names(names: Sequence[str]) -> List[str]:
    return [n.strip().lower() for n in names if n and n.strip()]


def resolve_metrics(metric_names: Optional[Sequence[str]] = None) -> List[Any]:
    _ensure_ragas_available()
    names = _normalize_metric_names(metric_names or DEFAULT_METRICS)
    metrics = []
    for name in names:
        metric = _METRIC_MAP.get(name)
        if metric is None:
            raise ValueError(f"지원하지 않는 RAGAS metric: {name}")
        metrics.append(metric)
    return metrics


def load_qa_dataset(path: Path, limit: Optional[int] = None) -> List[QAItem]:
    items: List[QAItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                QAItem(
                    id=data["id"],
                    question=data["question"],
                    answer=data.get("answer", ""),
                    source_ids=data.get("source_ids", []),
                )
            )
            if limit and len(items) >= limit:
                break
    return items


def _aggregate_scores(scores: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in scores:
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k)}


def _extract_summary(result: Any) -> Dict[str, float]:
    if hasattr(result, "scores"):
        return _aggregate_scores(result.scores)
    try:
        return dict(result)
    except Exception:
        return {}


def _build_ragas_dataset(rows: List[Dict[str, Any]]) -> "Dataset":
    _ensure_ragas_available()
    data = {
        "id": [],
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "reference": [],
    }
    for row in rows:
        data["id"].append(row.get("id", ""))
        data["question"].append(row.get("question", ""))
        data["answer"].append(row.get("answer", ""))
        data["contexts"].append(row.get("contexts", []))
        data["ground_truth"].append(row.get("ground_truth", ""))
        data["reference"].append(row.get("ground_truth", ""))
    return Dataset.from_dict(data)


def run_eval(
    qa_items: List[QAItem],
    *,
    host: str,
    ranker: str,
    top_k: int,
    sleep: float,
    include_empty: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in qa_items:
        res = call_query(host, item.question, ranker, top_k)
        sources = res.get("sources", [])
        contexts = [s.get("text", "") for s in sources if s.get("text")]
        answer = res.get("answer", "") or ""

        if not include_empty and (not answer or not contexts):
            rows.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item.answer,
                    "skipped": True,
                    "skip_reason": "empty_answer_or_context",
                }
            )
            continue

        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.answer,
                "skipped": False,
            }
        )
        if sleep:
            time.sleep(sleep)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", type=Path, required=True, help="QA JSONL 경로")
    ap.add_argument("--host", default=DEFAULT_BASE_URL, help="/query 서버 base URL")
    ap.add_argument("--ranker", default="none", help="리랭커 (none|llm|bge|llm-lite|auto)")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="평가 샘플 수 제한")
    ap.add_argument("--sleep", type=float, default=0.0, help="요청 간 대기 시간 (초)")
    ap.add_argument("--output", type=Path, required=True, help="RAGAS 결과 JSON 경로")
    ap.add_argument("--dump-samples", type=Path, default=None, help="RAGAS 입력 샘플 JSONL 저장 경로")
    ap.add_argument("--include-empty", action="store_true", help="빈 답변/컨텍스트도 포함")
    ap.add_argument("--llm-model", default=None, help="평가용 LLM 모델명")
    ap.add_argument("--llm-base-url", default=None, help="평가용 LLM base URL")
    ap.add_argument("--llm-api-key", default=None, help="평가용 LLM API 키")
    ap.add_argument("--emb-model", default=None, help="임베딩 모델명 (로컬 HuggingFace)")
    ap.add_argument("--emb-device", default=None, help="임베딩 디바이스 (cuda|cpu)")
    ap.add_argument("--metric", "metrics", action="append", default=[], help="RAGAS metric (복수 지정)")
    ap.add_argument("--batch-size", type=int, default=None, help="RAGAS 배치 크기")
    ap.add_argument("--no-progress", action="store_true", help="진행률 표시 비활성화")
    args = ap.parse_args()

    _ensure_ragas_available()

    qa_items = load_qa_dataset(args.qa_file, limit=args.limit)
    rows = run_eval(
        qa_items,
        host=args.host,
        ranker=args.ranker,
        top_k=args.top_k,
        sleep=args.sleep,
        include_empty=args.include_empty,
    )

    valid_rows = [r for r in rows if not r.get("skipped")]
    if not valid_rows:
        raise RuntimeError("RAGAS 평가 대상이 없습니다. (빈 답변/컨텍스트)")

    if args.llm_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = args.llm_api_key
    if args.llm_base_url and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = args.llm_base_url

    base_url = args.llm_base_url or os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or args.llm_api_key or "",
    )
    llm = llm_factory(model=args.llm_model or "gpt-4o-mini", client=client)
    embedding_model = args.emb_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device = args.emb_device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
    )
    metrics = resolve_metrics(args.metrics or None)
    dataset = _build_ragas_dataset(valid_rows)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=not args.no_progress,
        batch_size=args.batch_size,
    )

    summary = _extract_summary(result)
    payload = {
        "qa_file": str(args.qa_file),
        "num_total": len(rows),
        "num_evaluated": len(valid_rows),
        "num_skipped": len(rows) - len(valid_rows),
        "metrics": args.metrics or DEFAULT_METRICS,
        "summary": summary,
        "skipped": [r for r in rows if r.get("skipped")][:20],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dump_samples:
        args.dump_samples.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_samples.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"RAGAS 결과 저장: {args.output}")
    if args.dump_samples:
        print(f"샘플 저장: {args.dump_samples}")


if __name__ == "__main__":
    main()
