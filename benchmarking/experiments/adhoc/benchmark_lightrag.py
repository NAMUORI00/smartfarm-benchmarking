"""Benchmark LightRAG vs PathRAG on AgriQA dataset."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas.BaseSchemas import SourceDoc
from core.Services.Retrieval.LightRAG import (
    LightRAGIndexer,
    LightRAGHybridRetriever,
    QueryMode,
)


def load_corpus(corpus_path: Path, max_docs: int = 1000) -> list:
    """Load AgriQA corpus."""
    docs = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            data = json.loads(line)
            doc = SourceDoc(
                id=data['_id'],
                text=data['text'],
                metadata={'title': data.get('title', '')},
            )
            docs.append(doc)
    return docs


def load_queries(queries_path: Path) -> dict:
    """Load queries."""
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries[item['_id']] = item['text']
    return queries


def load_qrels(qrels_path: Path) -> dict:
    """Load relevance judgments."""
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, doc_id = parts[0], parts[1]
                rel = int(float(parts[2]))
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = rel
    return qrels


def evaluate_retriever(retriever, queries, qrels, k=10):
    """Evaluate a retriever on given queries."""
    hits = 0
    mrr = 0.0
    retrieval_counts = []

    for qid, q_text in queries.items():
        docs = retriever.search(q_text, k=k)
        doc_ids = [d.id for d in docs]
        retrieval_counts.append(len(doc_ids))

        if qid in qrels:
            relevant_docs = set(qrels[qid].keys())
            for i, doc_id in enumerate(doc_ids):
                if doc_id in relevant_docs:
                    hits += 1
                    mrr += 1.0 / (i + 1)
                    break

    n = len(queries)
    return {
        'hit_rate': hits / n if n > 0 else 0,
        'mrr': mrr / n if n > 0 else 0,
        'avg_retrieved': sum(retrieval_counts) / n if n > 0 else 0,
    }


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    corpus_path = data_dir / 'agriqa' / 'corpus.jsonl'
    queries_path = data_dir / 'agriqa' / 'queries.jsonl'
    qrels_path = data_dir / 'agriqa' / 'qrels' / 'test.tsv'

    print("Loading data...")
    docs = load_corpus(corpus_path, max_docs=10000)  # Full corpus
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    print(f"Loaded {len(docs)} documents, {len(queries)} queries")

    # Build LightRAG index
    print("\nBuilding LightRAG index...")
    start = time.time()
    indexer = LightRAGIndexer(
        embedding_dim=1000,  # Higher dim for TF-IDF
        chunk_size=512,
        use_llm_extraction=False,  # Use fast keyword extraction
    )
    index = indexer.index_documents(docs)
    index_time = time.time() - start
    print(f"Index built in {index_time:.1f}s")
    print(f"Index stats: {index.stats}")

    # Test all queries that have relevance judgments
    test_queries = {qid: queries[qid] for qid in qrels.keys() if qid in queries}
    print(f"\nEvaluating on {len(test_queries)} queries...")

    results = {}

    # Benchmark LightRAG modes
    for mode in [QueryMode.LOCAL, QueryMode.GLOBAL, QueryMode.HYBRID]:
        print(f"\nTesting LightRAG {mode.value} mode...")
        retriever = LightRAGHybridRetriever(
            index,
            mode=mode,
            entity_top_k=10,
            chunk_top_k=20,
        )

        start = time.time()
        metrics = evaluate_retriever(retriever, test_queries, qrels, k=10)
        elapsed = time.time() - start

        results[f'lightrag_{mode.value}'] = {
            **metrics,
            'time_sec': elapsed,
        }
        print(f"  Hit@10: {metrics['hit_rate']:.2%}")
        print(f"  MRR@10: {metrics['mrr']:.4f}")
        print(f"  Avg retrieved: {metrics['avg_retrieved']:.1f}")

    # Benchmark PathRAG (keyword mode) for comparison
    try:
        print("\nTesting PathRAG keyword mode...")
        from core.Models.Graph import SmartFarmGraph, GraphNode, GraphEdge
        from core.Services.Retrieval.PathRAG import PathRAGLtRetriever

        graph_path = data_dir / 'index' / 'smartfarm_graph_llm.json'
        if graph_path.exists():
            graph = SmartFarmGraph()
            with open(graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for n in data.get('nodes', []):
                node = GraphNode(
                    id=n['id'],
                    type=n['type'],
                    name=n['name'],
                    description=n.get('description'),
                    metadata=n.get('metadata', {}),
                )
                graph.add_node(node)

            for e in data.get('edges', []):
                edge = GraphEdge(
                    source=e['source'],
                    target=e['target'],
                    type=e['type'],
                    weight=e.get('weight', 1.0),
                )
                graph.add_edge(edge)

            retriever = PathRAGLtRetriever(
                graph,
                seed_match_mode='keyword',
                max_hops=3,
                threshold=0.1,
            )

            start = time.time()
            metrics = evaluate_retriever(retriever, test_queries, qrels, k=10)
            elapsed = time.time() - start

            results['pathrag_keyword'] = {
                **metrics,
                'time_sec': elapsed,
            }
            print(f"  Hit@10: {metrics['hit_rate']:.2%}")
            print(f"  MRR@10: {metrics['mrr']:.4f}")
            print(f"  Avg retrieved: {metrics['avg_retrieved']:.1f}")
    except Exception as e:
        print(f"  PathRAG benchmark skipped: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Hit@10: {metrics['hit_rate']:.2%}")
        print(f"  MRR@10: {metrics['mrr']:.4f}")
        print(f"  Avg retrieved: {metrics['avg_retrieved']:.1f}")
        print(f"  Time: {metrics['time_sec']:.2f}s")

    # Best performer
    best = max(results.items(), key=lambda x: x[1]['hit_rate'])
    print(f"\nBest performer: {best[0]} (Hit@10: {best[1]['hit_rate']:.2%})")

    # Save results
    output_path = Path('output/benchmark/lightrag_benchmark.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
