"""Paper-minimal evaluation runner (public datasets).

This runner is intentionally minimal and aligned with the paper intro goals:
  - Retrieval quality: Precision@K, Recall@K, MRR, nDCG@K
  - Answer quality (domain QA track): EM, Token-F1
  - Groundedness (optional): RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall)
  - System: latency (p50/p95), index build time, index size

Datasets:
  - agxqa: msu-ceco/agxqa_v1 (agriculture-domain extractive QA)
  - 2wiki: framolfese/2WikiMultihopQA (multi-hop reasoning; not agriculture-domain)

Usage examples:
  python -m benchmarking.experiments.paper_eval --dataset all --methods all --k 4 --out output/paper_eval.json
  python -m benchmarking.experiments.paper_eval --dataset agxqa --methods ours_full --with-ragas --ragas-max-queries 200
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from benchmarking.datasets import EvalSample, load_agxqa, load_twowiki_multihopqa
from benchmarking.metrics.qa_metrics import QAMetrics
from benchmarking.metrics.retrieval_metrics import RetrievalMetrics


DEFAULT_METHODS = ["bm25_only", "dense_only", "rrf", "trigraph_only", "ours_full"]
DEFAULT_DATASETS = ["agxqa", "2wiki"]


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
    """Percentile in milliseconds (linear interpolation; no numpy dependency)."""
    if not values_s:
        return 0.0
    xs = sorted(float(v) for v in values_s)
    if len(xs) == 1:
        return xs[0] * 1000.0

    rank = (len(xs) - 1) * (float(p) / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    value = xs[lo] * (1.0 - frac) + xs[hi] * frac
    return float(value * 1000.0)


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


def _chunk_docs(
    docs: Sequence[Tuple[str, str]],
    *,
    chunk_size: int,
    chunk_stride: int,
):
    from benchmarking.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    from core.Models.Schemas import SourceDoc
    from core.Services.Ingest.Chunking import sentence_split, window_chunks

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


def _build_dense_sparse_indices(
    docs,
    *,
    out_dir: Path,
    embed_model_id: Optional[str],
    need_dense: bool,
    need_sparse: bool,
) -> Tuple[Any, Any, Dict[str, Any]]:
    from benchmarking.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    # Heavy deps (faiss, sklearn) live inside these modules; import lazily.
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import BM25Store

    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {}

    dense = EmbeddingRetriever(model_id=embed_model_id or None, cache_size=0)
    if need_dense:
        t0 = time.perf_counter()
        dense.build(list(docs))
        stats["dense_build_s"] = float(time.perf_counter() - t0)

        dense_index_path = out_dir / "dense.faiss"
        dense_docs_path = out_dir / "dense_docs.jsonl"
        dense.save(str(dense_index_path), str(dense_docs_path))
        stats["dense_size_bytes"] = int(_dir_size_bytes(dense_index_path) + _dir_size_bytes(dense_docs_path))
    else:
        stats["dense_build_s"] = 0.0
        stats["dense_size_bytes"] = 0

    stats["dense_model_id"] = str(getattr(dense, "model_id", ""))
    stats["dense_dim"] = int(getattr(dense, "dim", 0))

    sparse = BM25Store()
    if need_sparse:
        t0 = time.perf_counter()
        sparse.index(list(docs))
        stats["sparse_build_s"] = float(time.perf_counter() - t0)
        sparse_path = out_dir / "sparse_bm25.pkl"
        sparse.save(str(sparse_path))
        stats["sparse_size_bytes"] = int(_dir_size_bytes(sparse_path))
    else:
        stats["sparse_build_s"] = 0.0
        stats["sparse_size_bytes"] = 0

    return dense, sparse, stats


def _build_trigraph_index(
    docs,
    *,
    out_dir: Path,
    embedder,
) -> Tuple[Any, Dict[str, Any]]:
    from benchmarking.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    from core.Config.Settings import settings
    from core.Services.Retrieval.TriGraph.Builder import build_trigraph_from_chunks
    from core.Services.Retrieval.TriGraph.Retriever import TriGraphConfig, TriGraphRetriever

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    build_trigraph_from_chunks(
        list(docs),
        embedder=embedder,
        out_dir=out_dir,
        chunk_size=int(settings.CHUNK_SIZE),
        chunk_stride=int(settings.CHUNK_STRIDE),
        corpus_path=str(out_dir),
    )
    build_s = float(time.perf_counter() - t0)

    stats = {
        "trigraph_build_s": build_s,
        "trigraph_size_bytes": int(_dir_size_bytes(out_dir)),
    }

    cfg = TriGraphConfig(
        index_dir=out_dir,
        entity_top_k=int(getattr(settings, "TRIGRAPH_ENTITY_TOP_K", 10)),
        entity_threshold=float(getattr(settings, "TRIGRAPH_ENTITY_THRESHOLD", 0.35)),
        max_iterations=int(getattr(settings, "TRIGRAPH_MAX_ITERATIONS", 3)),
        iteration_threshold=float(getattr(settings, "TRIGRAPH_ITERATION_THRESHOLD", 0.35)),
        top_k_sentence=int(getattr(settings, "TRIGRAPH_TOP_K_SENTENCE", 3)),
        max_active_entities=int(getattr(settings, "TRIGRAPH_MAX_ACTIVE_ENTITIES", 512)),
        max_candidate_chunks=int(getattr(settings, "TRIGRAPH_MAX_CANDIDATE_CHUNKS", 256)),
        use_ppr=bool(getattr(settings, "TRIGRAPH_USE_PPR", True)),
        ppr_damping=float(getattr(settings, "TRIGRAPH_PPR_DAMPING", 0.85)),
        ppr_iters=int(getattr(settings, "TRIGRAPH_PPR_ITERS", 16)),
    )

    docs_by_id = {d.id: d for d in docs}
    trigraph = TriGraphRetriever(cfg, docs_by_id=docs_by_id, embedder=embedder)
    return trigraph, stats


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
    from benchmarking.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    from core.Services.LLM import generate_json
    from core.Services.PromptTemplates import build_rag_answer_prompt

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

    for sample in samples:
        contexts = retriever.search(sample.question, k=int(k))
        prompt = build_rag_answer_prompt(sample.question, contexts)

        t0 = time.perf_counter()
        answer = ""
        err: Optional[str] = None
        try:
            answer = str((generate_json(prompt) or {}).get("answer") or "")
        except Exception as e:  # pragma: no cover - depends on LLMLITE host
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
        "mean_exact_match": sum([r["exact_match"] for r in qa_per_query]) / len(qa_per_query) if qa_per_query else 0.0,
        "mean_f1": sum([r["f1"] for r in qa_per_query]) / len(qa_per_query) if qa_per_query else 0.0,
    }

    retrieval_block["qa"] = qa_agg
    retrieval_block["generation_latency"] = {
        "n": int(len(gen_lat_s)),
        "p50_ms": _percentile_ms(gen_lat_s, 50),
        "p95_ms": _percentile_ms(gen_lat_s, 95),
    }
    retrieval_block["qa_per_query"] = qa_per_query
    return retrieval_block


def _resolve_ragas_llm(*, model: str, base_url: str, api_key: str):
    try:
        from langchain_openai import ChatOpenAI
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"langchain-openai is required for --with-ragas: {e}") from e

    llm = ChatOpenAI(
        model=str(model),
        temperature=0,
        api_key=str(api_key),
        base_url=str(base_url) if base_url else None,
        timeout=60,
        max_retries=1,
    )

    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(llm)
    except Exception:
        return llm


def _run_ragas_eval(
    *,
    samples: Sequence[EvalSample],
    k: int,
    rrf,
    ours_full,
    max_queries: int,
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Dict[str, float]]:
    """Run RAGAS (4 metrics only) for RRF vs Ours Full on a capped sample.

    We compute per-row scores and report mean values (standard reporting).
    """
    if rrf is None or ours_full is None:
        return {}

    try:
        from datasets import Dataset
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"datasets is required for --with-ragas: {e}") from e

    try:
        from ragas import evaluate
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"ragas is required for --with-ragas: {e}") from e

    # Metrics import (compat across ragas versions)
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    except Exception:  # pragma: no cover - version dependent
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        ragas_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

    evaluator_llm = _resolve_ragas_llm(model=model, base_url=base_url, api_key=api_key)

    from benchmarking.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    from core.Services.LLM import generate_json
    from core.Services.PromptTemplates import build_rag_answer_prompt

    capped = list(samples)[: int(max_queries)]

    def _build_dataset(method_retriever) -> Dataset:
        rows: List[Dict[str, Any]] = []
        for s in capped:
            contexts = method_retriever.search(s.question, k=int(k))
            ctx_texts = [str(c.text) for c in contexts]
            prompt = build_rag_answer_prompt(s.question, contexts)
            answer = ""
            try:
                answer = str((generate_json(prompt) or {}).get("answer") or "")
            except Exception:
                answer = ""
            rows.append(
                {
                    "question": s.question,
                    "answer": answer,
                    "contexts": ctx_texts,
                    "ground_truth": s.gold_answer,
                }
            )
        return Dataset.from_list(rows)

    out: Dict[str, Dict[str, float]] = {}
    for name, method_retriever in [("rrf", rrf), ("ours_full", ours_full)]:
        ds = _build_dataset(method_retriever)
        res = evaluate(
            dataset=ds,
            metrics=ragas_metrics,
            llm=evaluator_llm,
            raise_exceptions=False,
            show_progress=True,
        )

        scores: Dict[str, float] = {}
        try:
            df = res.to_pandas()
            for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if col in df.columns:
                    scores[col] = float(df[col].mean())
        except Exception:
            try:
                for k2, v2 in dict(res).items():  # type: ignore[arg-type]
                    scores[str(k2)] = float(v2)
            except Exception:
                scores = {}

        out[name] = scores

    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_methods(raw: str) -> List[str]:
    if not raw or raw.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or list(DEFAULT_METHODS)


def _parse_datasets(raw: str) -> List[str]:
    if not raw or raw.strip().lower() == "all":
        return list(DEFAULT_DATASETS)
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    return items or list(DEFAULT_DATASETS)


def _maybe_set_custom_prompts_dir() -> None:
    """For English public datasets, use an English RAG answer prompt by default.

    We intentionally do not touch the production prompt templates; we only set
    CUSTOM_PROMPTS_DIR for this process if it is not already set.
    """
    if os.getenv("CUSTOM_PROMPTS_DIR", "").strip():
        return
    prompt_dir = Path(__file__).resolve().parents[1] / "prompts" / "paper_eval_en"
    os.environ["CUSTOM_PROMPTS_DIR"] = str(prompt_dir)


def run_paper_eval(
    *,
    datasets: Sequence[str],
    methods: Sequence[str],
    k: int,
    out_path: Path,
    max_queries: int,
    seed: int,
    embed_model_id: Optional[str],
    with_ragas: bool,
    ragas_max_queries: int,
    ragas_model: str,
    ragas_base_url: str,
    ragas_api_key: str,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_set_custom_prompts_dir()

    from benchmarking.bootstrap import ensure_search_on_path

    ensure_search_on_path()
    from core.Config.Settings import settings

    run_meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": list(datasets),
        "k": int(k),
        "methods": list(methods),
        "chunk_size": int(settings.CHUNK_SIZE),
        "chunk_stride": int(settings.CHUNK_STRIDE),
        "embed_model_id": str(embed_model_id or getattr(settings, "EMBED_MODEL_ID", "")),
        "max_queries": int(max_queries),
        "seed": int(seed),
    }

    results: Dict[str, Any] = {"meta": run_meta, "datasets": {}}

    for ds_name in datasets:
        if ds_name == "agxqa":
            samples, corpus_docs = load_agxqa(split="test", max_queries=None, seed=seed)
            eval_mode = "qa"
        elif ds_name == "2wiki":
            samples, corpus_docs = load_twowiki_multihopqa(
                split="validation",
                max_queries=max_queries,
                seed=seed,
            )
            eval_mode = "retrieval"
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        chunks = _chunk_docs(
            corpus_docs,
            chunk_size=int(settings.CHUNK_SIZE),
            chunk_stride=int(settings.CHUNK_STRIDE),
        )

        dataset_out = out_path.parent / "paper_eval_artifacts" / ds_name

        need_dense_index = any(m in ("dense_only", "rrf", "ours_full") for m in methods)
        need_sparse_index = any(m in ("bm25_only", "rrf", "ours_full") for m in methods)

        dense, sparse, build_stats = _build_dense_sparse_indices(
            chunks,
            out_dir=dataset_out,
            embed_model_id=embed_model_id,
            need_dense=need_dense_index,
            need_sparse=need_sparse_index,
        )

        trigraph = None
        if any(m in ("trigraph_only", "ours_full") for m in methods):
            trigraph_dir = dataset_out / "trigraph_edge"
            trigraph, trigraph_stats = _build_trigraph_index(
                chunks,
                out_dir=trigraph_dir,
                embedder=dense,
            )
            build_stats.update(trigraph_stats)
        else:
            build_stats.update({"trigraph_build_s": 0.0, "trigraph_size_bytes": 0})

        from benchmarking.baselines.dense_only import DenseOnlyRetriever
        from benchmarking.baselines.ours_fusion import OursFullRetriever
        from benchmarking.baselines.rrf_hybrid import RRFHybridRetriever
        from benchmarking.baselines.sparse_only import SparseOnlyRetriever
        from benchmarking.baselines.trigraph_only import TriGraphOnlyRetriever

        from core.Services.Retrieval.Fusion import FusionConfig, TriGraphFusionRetriever

        retrievers: Dict[str, Any] = {
            "bm25_only": SparseOnlyRetriever(sparse),
            "dense_only": DenseOnlyRetriever(dense),
            "rrf": RRFHybridRetriever(dense, sparse),
            "trigraph_only": TriGraphOnlyRetriever(trigraph) if trigraph is not None else None,
            "ours_full": OursFullRetriever(
                TriGraphFusionRetriever(
                    dense,
                    sparse,
                    trigraph=trigraph,
                    config=FusionConfig(
                        rrf_k=int(getattr(settings, "FUSION_RRF_K", 60)),
                        use_dynamic_weights=bool(getattr(settings, "FUSION_DYNAMIC_WEIGHTS", True)),
                        base_dense=float(getattr(settings, "FUSION_BASE_DENSE", 0.45)),
                        base_sparse=float(getattr(settings, "FUSION_BASE_SPARSE", 0.35)),
                        base_trigraph=float(getattr(settings, "FUSION_BASE_TRIGRAPH", 0.20)),
                    ),
                )
            )
            if trigraph is not None
            else None,
        }

        k_values = [int(k), 10] if int(k) != 10 else [int(k)]
        out_block: Dict[str, Any] = {
            "mode": eval_mode,
            "n_queries": int(len(samples)),
            "n_docs": int(len(corpus_docs)),
            "n_chunks": int(len(chunks)),
            "index_build": build_stats,
            "results": {},
        }

        for m in methods:
            r = retrievers.get(m)
            if r is None:
                continue
            if eval_mode == "qa":
                out_block["results"][m] = _evaluate_with_generation(
                    method_name=m,
                    retriever=r,
                    samples=samples,
                    k=int(k),
                    k_values=k_values,
                )
            else:
                out_block["results"][m] = _evaluate_retrieval_only(
                    method_name=m,
                    retriever=r,
                    samples=samples,
                    k=int(k),
                    k_values=k_values,
                )

        if with_ragas and eval_mode == "qa":
            out_block["ragas"] = _run_ragas_eval(
                samples=samples,
                k=int(k),
                rrf=retrievers.get("rrf"),
                ours_full=retrievers.get("ours_full"),
                max_queries=int(ragas_max_queries),
                model=ragas_model,
                base_url=ragas_base_url,
                api_key=ragas_api_key,
            )

        results["datasets"][ds_name] = out_block

    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV outputs (paper tables)
    retrieval_rows: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []
    ragas_rows: List[Dict[str, Any]] = []

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

        ragas_block = dblock.get("ragas")
        if isinstance(ragas_block, dict):
            for mname, metrics in ragas_block.items():
                if not isinstance(metrics, dict):
                    continue
                ragas_rows.append(
                    {
                        "dataset": dname,
                        "method": mname,
                        "faithfulness": metrics.get("faithfulness", ""),
                        "answer_relevancy": metrics.get("answer_relevancy", ""),
                        "context_precision": metrics.get("context_precision", ""),
                        "context_recall": metrics.get("context_recall", ""),
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
    if ragas_rows:
        _write_csv(
            Path(str(base) + "_ragas.csv"),
            ragas_rows,
            fieldnames=["dataset", "method", "faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-minimal evaluation runner (public datasets)")
    parser.add_argument("--dataset", default="all", help="agxqa|2wiki|all or comma list")
    parser.add_argument("--methods", default="all", help="Comma list or 'all'")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default="output/paper_eval.json")

    # Dataset controls
    parser.add_argument("--max-queries", type=int, default=int(os.getenv("PAPER_EVAL_MAX_QUERIES", "1000")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("PAPER_EVAL_SEED", "42")))

    # Embedding model override (helps fast local runs on CPU)
    parser.add_argument("--embed-model-id", default=os.getenv("EMBED_MODEL_ID", ""), help="Override embedding model id")

    # Optional: RAGAS (LLM-as-a-Judge)
    parser.add_argument("--with-ragas", action="store_true", help="Run RAGAS (4 metrics) for RRF vs Ours Full (agxqa only)")
    parser.add_argument("--ragas-max-queries", type=int, default=int(os.getenv("RAGAS_MAX_QUERIES", "200")))
    parser.add_argument("--ragas-model", default=os.getenv("RAGAS_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    parser.add_argument("--ragas-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--ragas-api-key", default=os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", "")))

    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    methods = [m for m in methods if m in set(DEFAULT_METHODS)]
    if not methods:
        methods = list(DEFAULT_METHODS)

    datasets = _parse_datasets(args.dataset)
    datasets = [d for d in datasets if d in set(DEFAULT_DATASETS)]
    if not datasets:
        datasets = list(DEFAULT_DATASETS)

    embed_model_id = args.embed_model_id.strip() or None

    run_paper_eval(
        datasets=datasets,
        methods=methods,
        k=int(args.k),
        out_path=Path(args.out),
        max_queries=int(args.max_queries),
        seed=int(args.seed),
        embed_model_id=embed_model_id,
        with_ragas=bool(args.with_ragas),
        ragas_max_queries=int(args.ragas_max_queries),
        ragas_model=str(args.ragas_model),
        ragas_base_url=str(args.ragas_base_url),
        ragas_api_key=str(args.ragas_api_key),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
