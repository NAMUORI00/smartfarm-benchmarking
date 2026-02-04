from __future__ import annotations

"""2WikiMultiHopQA loader (public dataset; multi-hop reasoning).

References:
  - Original paper (COLING 2020):
    https://aclanthology.org/2020.coling-main.580/
  - Hugging Face (HotpotQA-format repack; parquet, no custom loader code):
    https://huggingface.co/datasets/framolfese/2WikiMultihopQA

Notes:
  - This dataset is not agriculture-domain; we use it to isolate multi-hop capability.
  - We build a *subset corpus* from the sampled queries' provided contexts:
      - each Wikipedia `title` becomes one "document"
      - document text = deduped sentences joined with newlines
  - Relevance is based on `supporting_facts.title` (multi-relevant).
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from .cache import resolve_hf_cache_dir
from .types import CorpusDoc, EvalSample


TWOWIKI_DATASET_ID = "framolfese/2WikiMultihopQA"


def _sha16(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_twowiki_multihopqa(
    *,
    split: str = "validation",
    max_queries: int | None = 1000,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> Tuple[List[EvalSample], List[CorpusDoc]]:
    """Load 2WikiMultiHopQA as (samples, corpus_docs).

    Args:
        split: HF split name (default: `validation`)
        max_queries: Optional cap on number of queries to load
        seed: Streaming shuffle seed (deterministic)
        cache_dir: Optional HF cache directory
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"`datasets` is required to load {TWOWIKI_DATASET_ID}: {e}") from e

    cache_path = resolve_hf_cache_dir(cache_dir)
    ds = load_dataset(TWOWIKI_DATASET_ID, split=split, streaming=True, cache_dir=str(cache_path))
    ds = ds.shuffle(seed=int(seed), buffer_size=10_000)

    title_to_sentences: Dict[str, List[str]] = {}
    samples: List[EvalSample] = []

    for row in ds:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        gold_answer = str(row.get("answer") or "").strip()

        if not qid:
            qid = f"2wiki_{len(samples)}"
        if not question:
            continue

        sf = row.get("supporting_facts") or {}
        sf_titles = [str(t).strip() for t in (sf.get("title") or []) if str(t).strip()]

        # Ensure corpus contains all supporting titles even if a context is missing.
        for t in sf_titles:
            title_to_sentences.setdefault(t, [])

        relevant_doc_ids = {(_sha16(t)) for t in sf_titles}
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_doc_ids}

        ctx = row.get("context") or {}
        ctx_titles = ctx.get("title") or []
        ctx_sentences = ctx.get("sentences") or []

        for title, sentences in zip(ctx_titles, ctx_sentences):
            title_s = str(title).strip()
            if not title_s:
                continue
            bucket = title_to_sentences.setdefault(title_s, [])
            for s in sentences or []:
                sent = str(s).strip()
                if sent:
                    bucket.append(sent)

        samples.append(
            EvalSample(
                qid=qid,
                question=question,
                relevant_doc_ids=relevant_doc_ids,
                relevance_scores=relevance_scores,
                gold_answer=gold_answer,
            )
        )

        if max_queries is not None and len(samples) >= int(max_queries):
            break

    corpus_docs: List[CorpusDoc] = []
    for title, sentences in title_to_sentences.items():
        seen = set()
        uniq: List[str] = []
        for s in sentences:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        body = "\n".join(uniq).strip()
        text = f"{title}\n{body}".strip() if body else title
        corpus_docs.append((_sha16(title), text))

    corpus_docs.sort(key=lambda x: x[0])
    return samples, corpus_docs

