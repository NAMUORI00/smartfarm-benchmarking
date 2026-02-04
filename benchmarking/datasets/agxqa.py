from __future__ import annotations

"""AgXQA loader (public dataset; agriculture domain).

References:
  - AgXQA benchmark repo:
    https://github.com/MSU-CECO/agxqa_benchmark_v1
  - Hugging Face dataset card:
    https://huggingface.co/datasets/msu-ceco/agxqa_v1

Notes:
  - AgXQA is SQuAD-style extractive QA (question + single context + answer spans).
  - For retrieval evaluation, we treat each unique `context` as a "document".
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from .cache import resolve_hf_cache_dir
from .types import CorpusDoc, EvalSample


AGXQA_DATASET_ID = "msu-ceco/agxqa_v1"


def _sha16(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_agxqa(
    *,
    split: str = "test",
    max_queries: int | None = None,
    seed: int | None = None,
    cache_dir: Path | None = None,
) -> Tuple[List[EvalSample], List[CorpusDoc]]:
    """Load AgXQA as (samples, corpus_docs).

    Args:
        split: HF split name (default: `test`)
        max_queries: Optional cap on number of queries to load
        seed: Optional shuffle seed (streaming shuffle; deterministic)
        cache_dir: Optional HF cache directory
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"`datasets` is required to load {AGXQA_DATASET_ID}: {e}") from e

    cache_path = resolve_hf_cache_dir(cache_dir)
    ds = load_dataset(AGXQA_DATASET_ID, split=split, streaming=True, cache_dir=str(cache_path))
    if seed is not None:
        ds = ds.shuffle(seed=int(seed), buffer_size=10_000)

    corpus_by_id: Dict[str, str] = {}
    samples: List[EvalSample] = []

    for row in ds:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "")

        if not question or not context.strip():
            continue

        doc_id = _sha16(context)
        corpus_by_id.setdefault(doc_id, context)

        answers = row.get("answers") or {}
        gold_answer = ""
        try:
            gold_answer = str((answers.get("text") or [""])[0]).strip()
        except Exception:
            gold_answer = ""

        if not qid:
            qid = f"agxqa_{len(samples)}"

        samples.append(
            EvalSample(
                qid=qid,
                question=question,
                relevant_doc_ids={doc_id},
                relevance_scores={doc_id: 1.0},
                gold_answer=gold_answer,
            )
        )

        if max_queries is not None and len(samples) >= int(max_queries):
            break

    corpus_docs: List[CorpusDoc] = sorted(list(corpus_by_id.items()), key=lambda x: x[0])
    return samples, corpus_docs

