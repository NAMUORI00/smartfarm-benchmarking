"""Paper-minimal evaluation runner.

This script is intentionally minimal and aligned with the paper intro goals:
  - Retrieval quality: Precision@K, Recall@K, MRR, nDCG@K
  - Answer quality (multihop track only): EM, Token-F1
  - System: latency (p50/p95), index build time, index size

Tracks:
  - multihop: domain multi-hop QA with gold answers + supporting facts
  - beir: public retrieval-only (local BEIR subset already vendored in repo)

Usage examples:
  python -m benchmarking.experiments.paper_eval --track multihop --methods all --k 4 --out output/paper_eval.json
  python -m benchmarking.experiments.paper_eval --track beir --methods bm25_only,dense_only,rrf --out output/paper_eval_beir.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path
from benchmarking.metrics.qa_metrics import QAMetrics
from benchmarking.metrics.retrieval_metrics import RetrievalMetrics

ensure_search_on_path()

from core.Config.Settings import settings
from core.Models.Schemas import SourceDoc
from core.Services.Ingest.Chunking import sentence_split, window_chunks
from core.Services.LLM import generate_json
from core.Services.PromptTemplates import build_rag_answer_prompt
from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store


DEFAULT_METHODS = ["bm25_only", "dense_only", "rrf"]


@dataclass(frozen=True)
class EvalSample:
    qid: str
    question: str
    relevant_doc_ids: Set[str]
    relevance_scores: Dict[str, float]
    gold_answer: str = ""


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _chunk_docs(
    docs: Sequence[Tuple[str, str]],
    *,
    chunk_size: int,
    chunk_stride: int,
) -> List[SourceDoc]:
    out: List[SourceDoc] = []
    for doc_id, text in docs:
        sents = sentence_split(text or "")
        chunks = window_chunks(sents, window=int(chunk_size), stride=int(chunk_stride))
        for i, chunk in enumerate(chunks):
            chunk_text = (chunk or "").strip()
            if not chunk_text:
                continue
            out.append(
                SourceDoc(
                    id=f"{doc_id}#c{i}",
                    text=chunk_text,
                    metadata={"doc_id": doc_id, "chunk_index": int(i)},
                )
            )
    return out


def _dedup_doc_ids_in_rank_order(chunk_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for cid in chunk_ids:
        doc_id = str(cid).split("#", 1)[0]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _percentile_ms(values_s: Sequence[float], p: float) -> float:
    if not values_s:
        return 0.0
    arr = np.array(list(values_s), dtype=np.float64) * 1000.0
    return float(np.percentile(arr, p))


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += int(p.stat().st_size)
    return int(total)


def _load_multihop_samples(data_dir: Path) -> Tuple[List[EvalSample], List[Tuple[str, str]]]:
    qa_path = data_dir / "multihop_gold.jsonl"
    corpus_path = data_dir / "causal_extraction_gold.jsonl"

    corpus_docs: List[Tuple[str, str]] = []
    for row in _iter_jsonl(corpus_path):
        doc_id = str(row.get("doc_id") or "").strip()
        text = str(row.get("text") or "")
        if doc_id and text.strip():
            corpus_docs.append((doc_id, text))

    samples: List[EvalSample] = []
    for row in _iter_jsonl(qa_path):
        qid = str(row.get("question_id") or "").strip()
        question = str(row.get("question") or "").strip()
        gold_answer = str(row.get("gold_answer") or "").strip()
        facts = row.get("gold_supporting_facts") or []

        relevant: Set[str] = set()
        for f in facts:
            doc_id = str((f or {}).get("doc_id") or "").strip()
            if doc_id:
                relevant.add(doc_id)

        if not qid or not question:
            continue

        # Binary relevance for ndcg (standard in many IR settings)
        rel_scores = {d: 1.0 for d in relevant}
        samples.append(
            EvalSample(
                qid=qid,
                question=question,
                relevant_doc_ids=relevant,
                relevance_scores=rel_scores,
                gold_answer=gold_answer,
            )
        )

    return samples, corpus_docs


def _load_beir_dataset(
    dataset_dir: Path,
    *,
    split: str,
    max_queries: int,
    doc_limit: int,
) -> Tuple[List[EvalSample], List[Tuple[str, str]]]:
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels: {qrels_path}")

    # qrels: query-id, corpus-id, score
    qrels: Dict[str, Dict[str, float]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header:
                header = False
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                score = float(score_s)
            except Exception:
                score = 0.0
            if not qid or not doc_id:
                continue
            qrels.setdefault(qid, {})[doc_id] = score

    queries: List[Tuple[str, str]] = []
    for row in _iter_jsonl(queries_path):
        qid = str(row.get("_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not qid or not text:
            continue
        if qid not in qrels:
            continue
        queries.append((qid, text))
        if len(queries) >= int(max_queries):
            break

    selected_qids = {qid for qid, _ in queries}
    needed_docs: Set[str] = set()
    for qid in selected_qids:
        needed_docs.update(qrels.get(qid, {}).keys())

    # Stream corpus and keep: (1) all needed docs; (2) extra docs until limit.
    corpus_docs: List[Tuple[str, str]] = []
    kept: Set[str] = set()

    for row in _iter_jsonl(corpus_path):
        doc_id = str(row.get("_id") or "").strip()
        if not doc_id or doc_id in kept:
            continue
        want = doc_id in needed_docs
        room = len(corpus_docs) < int(doc_limit)
        if not want and not room:
            continue
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        full = (title + "\n" + text).strip() if title else text
        if not full:
            continue
        corpus_docs.append((doc_id, full))
        kept.add(doc_id)
        if len(corpus_docs) >= int(doc_limit) and needed_docs.issubset(kept):
            break

    samples: List[EvalSample] = []
    for qid, question in queries:
        rel_map = qrels.get(qid, {})
        relevant = set(rel_map.keys())
        # Filter relevance to docs we actually kept (avoids inflated denominators on subset runs)
        relevant = set([d for d in relevant if d in kept])
        rel_scores = {d: float(rel_map.get(d, 0.0)) for d in relevant}
        samples.append(
            EvalSample(
                qid=qid,
                question=question,
                relevant_doc_ids=relevant,
                relevance_scores=rel_scores,
            )
        )

    return samples, corpus_docs


def _build_dense_sparse_indices(
    docs: List[SourceDoc],
    *,
    out_dir: Path,
    embed_model_id: Optional[str],
    need_dense: bool,
    need_sparse: bool,
) -> Tuple[EmbeddingRetriever, BM25Store, Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {}

    # Dense
    dense = EmbeddingRetriever(model_id=embed_model_id or None, cache_size=0)
    if need_dense:
        t0 = time.perf_counter()
        dense.build(docs)
        stats["dense_build_s"] = float(time.perf_counter() - t0)

        dense_index_path = out_dir / "dense.faiss"
        dense_docs_path = out_dir / "dense_docs.jsonl"
        dense.save(str(dense_index_path), str(dense_docs_path))
        stats["dense_size_bytes"] = int(_dir_size_bytes(dense_index_path) + _dir_size_bytes(dense_docs_path))
        stats["dense_model_id"] = str(getattr(dense, "model_id", ""))
        stats["dense_dim"] = int(getattr(dense, "dim", 0))
    else:
        stats["dense_build_s"] = 0.0
        stats["dense_size_bytes"] = 0
        stats["dense_model_id"] = str(getattr(dense, "model_id", ""))
        stats["dense_dim"] = int(getattr(dense, "dim", 0))

    # Sparse (BM25)
    sparse = BM25Store()
    if need_sparse:
        t0 = time.perf_counter()
        sparse.index(docs)
        stats["sparse_build_s"] = float(time.perf_counter() - t0)
        sparse_path = out_dir / "sparse_bm25.pkl"
        sparse.save(str(sparse_path))
        stats["sparse_size_bytes"] = int(_dir_size_bytes(sparse_path))
    else:
        stats["sparse_build_s"] = 0.0
        stats["sparse_size_bytes"] = 0

    return dense, sparse, stats


def _evaluate_retrieval_only(
    *,
    method_name: str,
    retriever,
    samples: Sequence[EvalSample],
    k: int,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    metrics = RetrievalMetrics(k_values=list(k_values))
    per_query: List[Dict[str, Any]] = []
    lat_s: List[float] = []

    for s in samples:
        t0 = time.perf_counter()
        docs = retriever.search(s.question, k=int(k))
        dt = time.perf_counter() - t0
        lat_s.append(float(dt))

        retrieved_doc_ids = _dedup_doc_ids_in_rank_order([d.id for d in docs])
        m = metrics.compute_all(
            retrieved=retrieved_doc_ids,
            relevant=set(s.relevant_doc_ids),
            relevance_scores=dict(s.relevance_scores),
        )
        per_query.append(
            {
                "qid": s.qid,
                "retrieved": retrieved_doc_ids,
                "relevant": sorted(list(s.relevant_doc_ids)),
                "metrics": m,
                "latency_ms": float(dt * 1000.0),
            }
        )

    agg = metrics.aggregate([r["metrics"] for r in per_query])
    return {
        "method": method_name,
        "retrieval": agg,
        "latency": {
            "n": int(len(lat_s)),
            "p50_ms": _percentile_ms(lat_s, 50),
            "p95_ms": _percentile_ms(lat_s, 95),
        },
        "per_query": per_query,
    }


def _evaluate_with_generation(
    *,
    method_name: str,
    retriever,
    samples: Sequence[EvalSample],
    k: int,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    retrieval_block = _evaluate_retrieval_only(
        method_name=method_name,
        retriever=retriever,
        samples=samples,
        k=k,
        k_values=k_values,
    )

    qa = QAMetrics()
    qa_per_query: List[Dict[str, Any]] = []
    gen_lat_s: List[float] = []

    for row, sample in zip(retrieval_block["per_query"], samples):
        # Re-run retrieval to get contexts (kept separate to avoid storing full texts in JSON)
        # This keeps output smaller while ensuring contexts match the evaluated retriever.
        contexts = retriever.search(sample.question, k=int(k))
        prompt = build_rag_answer_prompt(sample.question, contexts)
        t0 = time.perf_counter()
        answer = ""
        err: Optional[str] = None
        try:
            answer = str((generate_json(prompt) or {}).get("answer") or "")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        gen_lat_s.append(float(dt))

        qa_metrics = qa.compute_all(prediction=answer, reference=sample.gold_answer)
        qa_per_query.append(
            {
                "qid": sample.qid,
                "exact_match": qa_metrics["exact_match"],
                "f1": qa_metrics["f1"],
                "gen_latency_ms": float(dt * 1000.0),
                "error": err,
            }
        )

    qa_agg = {
        "mean_exact_match": float(np.mean([r["exact_match"] for r in qa_per_query])) if qa_per_query else 0.0,
        "mean_f1": float(np.mean([r["f1"] for r in qa_per_query])) if qa_per_query else 0.0,
    }

    retrieval_block["qa"] = qa_agg
    retrieval_block["generation_latency"] = {
        "n": int(len(gen_lat_s)),
        "p50_ms": _percentile_ms(gen_lat_s, 50),
        "p95_ms": _percentile_ms(gen_lat_s, 95),
    }
    retrieval_block["qa_per_query"] = qa_per_query
    return retrieval_block


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def run_paper_eval(
    *,
    track: str,
    methods: List[str],
    k: int,
    out_path: Path,
    beir_max_queries: int,
    beir_doc_limit: int,
    beir_split: str,
    embed_model_id: Optional[str],
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parents[1] / "data"

    run_meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": track,
        "k": int(k),
        "methods": list(methods),
        "chunk_size": int(settings.CHUNK_SIZE),
        "chunk_stride": int(settings.CHUNK_STRIDE),
        "embed_model_id": str(embed_model_id or getattr(settings, "EMBED_MODEL_ID", "")),
    }

    results: Dict[str, Any] = {"meta": run_meta, "datasets": {}}

    if track == "multihop":
        samples, corpus_docs = _load_multihop_samples(data_dir)
        chunks = _chunk_docs(
            corpus_docs,
            chunk_size=int(settings.CHUNK_SIZE),
            chunk_stride=int(settings.CHUNK_STRIDE),
        )

        dataset_out = out_path.parent / "paper_eval_artifacts" / "multihop"
        dense, sparse, build_stats = _build_dense_sparse_indices(
            chunks,
            out_dir=dataset_out,
            embed_model_id=embed_model_id,
            need_dense=any(m in ("dense_only", "rrf") for m in methods),
            need_sparse=any(m in ("bm25_only", "rrf") for m in methods),
        )

        from benchmarking.baselines.dense_only import DenseOnlyRetriever
        from benchmarking.baselines.sparse_only import SparseOnlyRetriever
        from benchmarking.baselines.rrf_hybrid import RRFHybridRetriever

        retrievers: Dict[str, Any] = {
            "bm25_only": SparseOnlyRetriever(sparse),
            "dense_only": DenseOnlyRetriever(dense),
            "rrf": RRFHybridRetriever(dense, sparse),
        }

        k_values = [int(k), 10] if int(k) != 10 else [int(k)]
        out_block: Dict[str, Any] = {
            "n_queries": int(len(samples)),
            "n_docs": int(len(corpus_docs)),
            "n_chunks": int(len(chunks)),
            "index_build": build_stats,
            "results": {},
        }

        for m in methods:
            if m not in retrievers:
                continue
            out_block["results"][m] = _evaluate_with_generation(
                method_name=m,
                retriever=retrievers[m],
                samples=samples,
                k=int(k),
                k_values=k_values,
            )

        results["datasets"]["multihop"] = out_block

    elif track == "beir":
        beir_root = data_dir / "beir"
        datasets = [
            ("quora", beir_root / "quora"),
            ("trec-covid", beir_root / "trec-covid"),
        ]

        for name, dpath in datasets:
            samples, corpus_docs = _load_beir_dataset(
                dpath,
                split=beir_split,
                max_queries=beir_max_queries,
                doc_limit=beir_doc_limit,
            )
            chunks = _chunk_docs(
                corpus_docs,
                chunk_size=int(settings.CHUNK_SIZE),
                chunk_stride=int(settings.CHUNK_STRIDE),
            )
            dataset_out = out_path.parent / "paper_eval_artifacts" / f"beir_{name}"
            dense, sparse, build_stats = _build_dense_sparse_indices(
                chunks,
                out_dir=dataset_out,
                embed_model_id=embed_model_id,
                need_dense=any(m in ("dense_only", "rrf") for m in methods),
                need_sparse=any(m in ("bm25_only", "rrf") for m in methods),
            )

            from benchmarking.baselines.dense_only import DenseOnlyRetriever
            from benchmarking.baselines.sparse_only import SparseOnlyRetriever
            from benchmarking.baselines.rrf_hybrid import RRFHybridRetriever

            retrievers: Dict[str, Any] = {
                "bm25_only": SparseOnlyRetriever(sparse),
                "dense_only": DenseOnlyRetriever(dense),
                "rrf": RRFHybridRetriever(dense, sparse),
            }

            k_values = [int(k), 10] if int(k) != 10 else [int(k)]
            out_block: Dict[str, Any] = {
                "split": beir_split,
                "n_queries": int(len(samples)),
                "n_docs": int(len(corpus_docs)),
                "n_chunks": int(len(chunks)),
                "index_build": build_stats,
                "results": {},
            }

            for m in methods:
                if m not in retrievers:
                    continue
                out_block["results"][m] = _evaluate_retrieval_only(
                    method_name=m,
                    retriever=retrievers[m],
                    samples=samples,
                    k=int(k),
                    k_values=k_values,
                )

            results["datasets"][f"beir_{name}"] = out_block
    else:
        raise ValueError(f"Unknown track: {track}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV outputs (paper tables)
    retrieval_rows: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []

    for dname, dblock in results["datasets"].items():
        for mname, mblock in (dblock.get("results") or {}).items():
            r = mblock.get("retrieval") or {}
            retrieval_rows.append(
                {
                    "dataset": dname,
                    "method": mname,
                    "mean_precision@4": r.get("mean_precision@4", ""),
                    "mean_recall@4": r.get("mean_recall@4", ""),
                    "mean_mrr": r.get("mean_mrr", ""),
                    "mean_ndcg@10": r.get("mean_ndcg@10", ""),
                }
            )
            if "qa" in mblock:
                qa_block = mblock.get("qa") or {}
                qa_rows.append(
                    {
                        "dataset": dname,
                        "method": mname,
                        "mean_exact_match": qa_block.get("mean_exact_match", ""),
                        "mean_f1": qa_block.get("mean_f1", ""),
                    }
                )

    base = out_path.with_suffix("")
    _write_csv(
        Path(str(base) + "_retrieval.csv"),
        retrieval_rows,
        fieldnames=["dataset", "method", "mean_precision@4", "mean_recall@4", "mean_mrr", "mean_ndcg@10"],
    )
    if qa_rows:
        _write_csv(
            Path(str(base) + "_qa.csv"),
            qa_rows,
            fieldnames=["dataset", "method", "mean_exact_match", "mean_f1"],
        )

    return results


def _parse_methods(raw: str) -> List[str]:
    if not raw or raw.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or list(DEFAULT_METHODS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-minimal evaluation runner")
    parser.add_argument("--track", required=True, choices=["multihop", "beir"])
    parser.add_argument("--methods", default="all", help="Comma list or 'all'")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default="output/paper_eval.json")

    # BEIR controls (local subset; do not download)
    parser.add_argument("--beir-split", default="test", choices=["dev", "test"])
    parser.add_argument("--beir-max-queries", type=int, default=int(os.getenv("BEIR_MAX_QUERIES", "50")))
    parser.add_argument("--beir-doc-limit", type=int, default=int(os.getenv("BEIR_DOC_LIMIT", "5000")))

    # Embedding model override (helps fast local runs on CPU)
    parser.add_argument("--embed-model-id", default=os.getenv("EMBED_MODEL_ID", ""), help="Override embedding model id")

    args = parser.parse_args()

    methods = _parse_methods(args.methods)

    # For now (commit 2), only support bm25/dense/rrf. Tri-Graph baselines come in the next commit.
    supported = set(DEFAULT_METHODS)
    methods = [m for m in methods if m in supported]
    if not methods:
        methods = list(DEFAULT_METHODS)

    embed_model_id = args.embed_model_id.strip() or None

    run_paper_eval(
        track=str(args.track),
        methods=methods,
        k=int(args.k),
        out_path=Path(args.out),
        beir_max_queries=int(args.beir_max_queries),
        beir_doc_limit=int(args.beir_doc_limit),
        beir_split=str(args.beir_split),
        embed_model_id=embed_model_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
