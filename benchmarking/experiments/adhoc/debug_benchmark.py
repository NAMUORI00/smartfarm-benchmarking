"""Debug benchmark results."""

from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas.BaseSchemas import SourceDoc
from core.Services.Retrieval.LightRAG.Indexer import LightRAGIndexer
from core.Services.Retrieval.LightRAG.HybridRetriever import LightRAGHybridRetriever, QueryMode


def main():
    # Load sample data
    corpus_path = PROJECT_ROOT / "benchmarking" / "data" / "agriqa" / "corpus.jsonl"
    queries_path = PROJECT_ROOT / "benchmarking" / "data" / "agriqa" / "queries.jsonl"
    qrels_path = PROJECT_ROOT / "benchmarking" / "data" / "agriqa" / "qrels" / "test.tsv"

    # Load corpus
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100:  # Only first 100 docs
                break
            data = json.loads(line)
            docs.append(SourceDoc(
                id=data["_id"],
                text=data["text"],
                metadata={"title": data.get("title", "")},
            ))

    print(f"Loaded {len(docs)} docs")
    print(f"Sample doc IDs: {[d.id for d in docs[:5]]}")

    # Load queries
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only first 10 queries
                break
            data = json.loads(line)
            queries[data["_id"]] = data["text"]

    print(f"\nLoaded {len(queries)} queries")
    print(f"Sample queries: {list(queries.items())[:2]}")

    # Load qrels
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, doc_id = parts[0], parts[2]
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(doc_id)

    print(f"\nLoaded qrels for {len(qrels)} queries")
    # Show sample qrels
    for qid, rel_docs in list(qrels.items())[:3]:
        print(f"  Query {qid}: relevant docs = {rel_docs}")

    # Build index
    print("\nBuilding index...")
    indexer = LightRAGIndexer()
    index = indexer.index_documents(docs)
    print(f"Index stats: {index.stats}")

    # Test a single query
    print("\n" + "=" * 60)
    print("Testing single query...")

    for qid, query in list(queries.items())[:3]:
        if qid not in qrels:
            print(f"  Query {qid} not in qrels, skipping")
            continue

        print(f"\nQuery {qid}: '{query[:50]}...'")
        print(f"  Relevant docs: {qrels[qid]}")

        retriever = LightRAGHybridRetriever(
            index=index,
            mode=QueryMode.HYBRID,
            chunk_threshold=0.0,  # No threshold
        )

        results = retriever.search(query, k=10)
        result_ids = [doc.id for doc in results]
        print(f"  Results: {result_ids[:5]}...")

        # Check if any hit
        hits = [rid for rid in result_ids if rid in qrels[qid]]
        print(f"  Hits: {hits}")


if __name__ == "__main__":
    main()
