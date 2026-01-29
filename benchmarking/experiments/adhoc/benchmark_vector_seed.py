"""Benchmark vector vs keyword seed matching for PathRAG."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()


def simple_vectorize(texts, vocab=None):
    """Simple TF-IDF style vectorization for testing."""
    # Build vocabulary if not provided
    if vocab is None:
        vocab = set()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            vocab.update(words)
        vocab = sorted(vocab)

    vocab_idx = {w: i for i, w in enumerate(vocab)}

    # Vectorize
    vectors = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        word_counts = Counter(words)
        vec = np.zeros(len(vocab))
        for word, count in word_counts.items():
            if word in vocab_idx:
                vec[vocab_idx[word]] = count
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)

    return np.array(vectors), vocab


def match_seeds_vector_simple(query, node_vectors, node_ids, vocab, top_k=10, threshold=0.3):
    """Simple vector seed matching."""
    # Vectorize query
    query_vec, _ = simple_vectorize([query], vocab)
    query_vec = query_vec[0]

    # Compute cosine similarity
    similarities = np.dot(node_vectors, query_vec)

    # Get top-K above threshold
    top_indices = np.argsort(similarities)[::-1][:top_k]

    matched_ids = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            matched_ids.append(node_ids[idx])

    return matched_ids, similarities


def main():
    from core.Models.Graph import SmartFarmGraph, GraphNode, GraphEdge
    from core.Services.Retrieval.PathRAG import PathRAGLtRetriever
    from core.Services.Retrieval.PathScoring import bfs_weighted_paths, balanced_path_selection

    # Load LLM-based graph (has more concept nodes)
    graph_path = Path('data/index/smartfarm_graph_llm.json')
    if not graph_path.exists():
        graph_path = Path('data/index/smartfarm_graph.json')
    print(f'Loading graph from: {graph_path}')

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
            embedding=n.get('embedding')
        )
        graph.add_node(node)

    for e in data.get('edges', []):
        edge = GraphEdge(
            source=e['source'],
            target=e['target'],
            type=e['type'],
            weight=e.get('weight', 1.0),
            metadata=e.get('metadata', {})
        )
        graph.add_edge(edge)

    print(f'Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges')

    # Build simple embeddings for all concept nodes
    concept_nodes = [n for n in graph.nodes.values() if n.type != 'practice']
    print(f'Concept nodes: {len(concept_nodes)}')

    # Build texts and vocabulary
    node_texts = []
    node_ids = []
    for node in concept_nodes:
        text = node.name or ''
        if node.description:
            text = f'{text} {node.description}'
        node_texts.append(text.strip() or node.id)
        node_ids.append(node.id)

    # Build vectors
    print('Building node embeddings...')
    node_vectors, vocab = simple_vectorize(node_texts)
    print(f'Vocabulary size: {len(vocab)}')
    print(f'Node vectors shape: {node_vectors.shape}')

    # Load queries and qrels
    queries_path = Path('data/agriqa/queries.jsonl')
    qrels_path = Path('data/agriqa/qrels/test.tsv')

    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries[item['_id']] = item['text']

    print(f'Loaded {len(queries)} queries')

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

    print(f'Loaded qrels for {len(qrels)} queries')

    results = {}

    # Test keyword mode
    print('\nTesting keyword mode...')
    retriever_kw = PathRAGLtRetriever(
        graph,
        seed_match_mode='keyword',
        max_hops=3,
        threshold=0.1,
        alpha=0.8,
        max_results=15
    )

    hits_kw = 0
    mrr_kw = 0.0
    seed_counts_kw = []
    retrieval_counts_kw = []

    start_time = time.time()
    test_queries = list(queries.items())[:100]

    for qid, q_text in test_queries:
        seeds = retriever_kw._match_seeds(q_text)
        seed_counts_kw.append(len(seeds))

        docs = retriever_kw.search(q_text, k=10)
        doc_ids = [d.id for d in docs]
        retrieval_counts_kw.append(len(doc_ids))

        if qid in qrels:
            relevant_docs = set(qrels[qid].keys())
            for i, doc_id in enumerate(doc_ids):
                if doc_id in relevant_docs:
                    hits_kw += 1
                    mrr_kw += 1.0 / (i + 1)
                    break

    elapsed_kw = time.time() - start_time

    results['keyword'] = {
        'hit_rate': hits_kw / len(test_queries),
        'mrr': mrr_kw / len(test_queries),
        'avg_seeds': sum(seed_counts_kw) / len(seed_counts_kw),
        'avg_retrieved': sum(retrieval_counts_kw) / len(retrieval_counts_kw),
        'time_sec': elapsed_kw
    }
    print(f'  Hit@10: {results["keyword"]["hit_rate"]:.2%}')
    print(f'  MRR@10: {results["keyword"]["mrr"]:.4f}')
    print(f'  Avg seeds: {results["keyword"]["avg_seeds"]:.1f}')
    print(f'  Avg retrieved: {results["keyword"]["avg_retrieved"]:.1f}')

    # Test vector mode (using simple implementation)
    print('\nTesting vector mode (simple TF-IDF)...')

    hits_vec = 0
    mrr_vec = 0.0
    seed_counts_vec = []
    retrieval_counts_vec = []

    start_time = time.time()

    practice_ids = {nid for nid, node in graph.nodes.items() if node.type == 'practice'}

    for qid, q_text in test_queries:
        # Use vector matching for seeds
        matched_ids, sims = match_seeds_vector_simple(
            q_text, node_vectors, node_ids, vocab, top_k=10, threshold=0.1
        )
        seed_counts_vec.append(len(matched_ids))

        # Convert to nodes
        seed_nodes = [graph.nodes[mid] for mid in matched_ids if mid in graph.nodes]

        # Skip if no seeds
        if not seed_nodes:
            retrieval_counts_vec.append(0)
            continue

        # Manual search with vector seeds
        scores = bfs_weighted_paths(
            adjacency=graph.adjacency,
            reverse_adjacency=graph.reverse_adjacency,
            sources=[s.id for s in seed_nodes],
            practice_node_ids=practice_ids,
            max_hops=3,
            threshold=0.1,
            alpha=0.8,
        )

        if not scores:
            retrieval_counts_vec.append(0)
            continue

        selected = balanced_path_selection(scores, max_results=10)

        doc_ids = [node_id for node_id, score in selected]
        retrieval_counts_vec.append(len(doc_ids))

        if qid in qrels:
            relevant_docs = set(qrels[qid].keys())
            for i, doc_id in enumerate(doc_ids):
                if doc_id in relevant_docs:
                    hits_vec += 1
                    mrr_vec += 1.0 / (i + 1)
                    break

    elapsed_vec = time.time() - start_time

    results['vector'] = {
        'hit_rate': hits_vec / len(test_queries),
        'mrr': mrr_vec / len(test_queries),
        'avg_seeds': sum(seed_counts_vec) / len(seed_counts_vec),
        'avg_retrieved': sum(retrieval_counts_vec) / len(retrieval_counts_vec),
        'time_sec': elapsed_vec
    }
    print(f'  Hit@10: {results["vector"]["hit_rate"]:.2%}')
    print(f'  MRR@10: {results["vector"]["mrr"]:.4f}')
    print(f'  Avg seeds: {results["vector"]["avg_seeds"]:.1f}')
    print(f'  Avg retrieved: {results["vector"]["avg_retrieved"]:.1f}')

    # Compare
    print('\n=== COMPARISON ===')
    kw = results['keyword']
    vec = results['vector']
    print(f'Hit Rate: keyword={kw["hit_rate"]:.2%} vs vector={vec["hit_rate"]:.2%}')
    print(f'MRR: keyword={kw["mrr"]:.4f} vs vector={vec["mrr"]:.4f}')
    print(f'Avg Seeds: keyword={kw["avg_seeds"]:.1f} vs vector={vec["avg_seeds"]:.1f}')
    print(f'Avg Retrieved: keyword={kw["avg_retrieved"]:.1f} vs vector={vec["avg_retrieved"]:.1f}')

    if vec['avg_seeds'] > kw['avg_seeds']:
        improvement = (vec['avg_seeds'] - kw['avg_seeds']) / max(kw['avg_seeds'], 0.001) * 100
        print(f'\nVector mode finds {improvement:.0f}% more seeds!')

    if vec['hit_rate'] > kw['hit_rate']:
        improvement = (vec['hit_rate'] - kw['hit_rate']) / max(kw['hit_rate'], 0.001) * 100
        print(f'Vector mode improves hit rate by {improvement:.1f}%')
    elif vec['hit_rate'] < kw['hit_rate']:
        decline = (kw['hit_rate'] - vec['hit_rate']) / max(kw['hit_rate'], 0.001) * 100
        print(f'Keyword mode better by {decline:.1f}%')
    else:
        print('Both modes perform equally')

    # Save results
    output_path = Path('output/benchmark/vector_seed_benchmark.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
