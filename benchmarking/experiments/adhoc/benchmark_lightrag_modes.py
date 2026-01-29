"""Benchmark LightRAG query modes on AgriQA corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas.BaseSchemas import SourceDoc
from core.Services.Retrieval.LightRAG.Indexer import LightRAGIndexer
from core.Services.Retrieval.LightRAG.HybridRetriever import LightRAGHybridRetriever, QueryMode


def load_corpus(path: Path, max_docs: int = 10000) -> list[SourceDoc]:
    """Load AgriQA corpus."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            data = json.loads(line)
            docs.append(SourceDoc(
                id=data["_id"],
                text=data["text"],
                metadata={"title": data.get("title", "")},
            ))
    return docs


def load_qrels(path: Path) -> dict[str, set[str]]:
    """Load qrels (query relevance judgments).

    Format: query-id\tcorpus-id\tscore
    """
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, doc_id = parts[0], parts[1]  # parts[1] is corpus-id
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(doc_id)
    return qrels


def load_queries(path: Path) -> dict[str, str]:
    """Load queries."""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries[data["_id"]] = data["text"]
    return queries


def evaluate_retriever(retriever: LightRAGHybridRetriever, queries: dict, qrels: dict, k: int = 10) -> dict:
    """Evaluate retriever on queries."""
    total = 0
    hits = 0
    mrr_sum = 0.0

    for qid, query in queries.items():
        if qid not in qrels:
            continue

        total += 1
        relevant = qrels[qid]

        try:
            results = retriever.search(query, k=k)
            result_ids = [doc.id for doc in results]
        except Exception as e:
            print(f"  [Error] Query {qid}: {e}")
            continue

        # Hit@K
        if any(rid in relevant for rid in result_ids):
            hits += 1

        # MRR
        for i, rid in enumerate(result_ids):
            if rid in relevant:
                mrr_sum += 1.0 / (i + 1)
                break

    return {
        "total": total,
        "hits": hits,
        "hit_rate": hits / total * 100 if total > 0 else 0,
        "mrr": mrr_sum / total if total > 0 else 0,
    }


def main():
    print("=" * 70)
    print("LightRAG Query Mode Benchmark (AgriQA Corpus)")
    print("=" * 70)

    # Paths
    corpus_path = PROJECT_ROOT / "data" / "agriqa" / "corpus.jsonl"
    queries_path = PROJECT_ROOT / "data" / "agriqa" / "queries.jsonl"
    qrels_path = PROJECT_ROOT / "data" / "agriqa" / "qrels" / "test.tsv"

    # Load data
    print("\n[1] Loading data...")
    docs = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    print(f"    Corpus: {len(docs)} documents")
    print(f"    Queries: {len(queries)}")
    print(f"    Qrels: {len(qrels)} queries with relevance judgments")

    # Build index
    print("\n[2] Building LightRAG index...")
    indexer = LightRAGIndexer()
    index = indexer.index_documents(docs)
    print(f"    Index stats: {index.stats}")

    # Test different modes
    modes = [
        QueryMode.GLOBAL,
        QueryMode.LOCAL,
        QueryMode.HYBRID,
        QueryMode.PATHRAG,
        QueryMode.PATHRAG_HYBRID,
    ]

    results = {}

    print("\n[3] Evaluating query modes (Hit@10, MRR)...\n")
    for mode in modes:
        print(f"    Testing {mode.value.upper()}...")
        retriever = LightRAGHybridRetriever(
            index=index,
            mode=mode,
            entity_top_k=10,
            chunk_top_k=20,
            entity_threshold=0.2,
            chunk_threshold=0.1,
        )
        metrics = evaluate_retriever(retriever, queries, qrels, k=10)
        results[mode.value] = metrics
        print(f"      Hit@10: {metrics['hit_rate']:.2f}% | MRR: {metrics['mrr']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<12} | {'Hit@10':>10} | {'MRR':>8} | {'Hits/Total':>12}")
    print("-" * 50)
    for mode, m in results.items():
        print(f"{mode.upper():<12} | {m['hit_rate']:>9.2f}% | {m['mrr']:>8.4f} | {m['hits']}/{m['total']}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
