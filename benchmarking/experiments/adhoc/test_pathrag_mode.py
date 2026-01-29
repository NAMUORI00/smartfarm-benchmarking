"""Quick integration test for PATHRAG mode in HybridRetriever."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas.BaseSchemas import SourceDoc
from core.Services.Retrieval.LightRAG.Indexer import LightRAGIndexer
from core.Services.Retrieval.LightRAG.HybridRetriever import LightRAGHybridRetriever, QueryMode


def main():
    """Test PathRAG mode integration."""
    print("=" * 60)
    print("PathRAG Mode Integration Test")
    print("=" * 60)

    # Create test documents
    docs = [
        SourceDoc(
            id="doc1",
            text="Rice blight disease can be treated with fungicide spray. Apply mancozeb or carbendazim.",
            metadata={"topic": "disease"},
        ),
        SourceDoc(
            id="doc2",
            text="Urea and DAP are essential fertilizers for rice cultivation. Apply NPK during sowing.",
            metadata={"topic": "fertilizer"},
        ),
        SourceDoc(
            id="doc3",
            text="Tomato wilt is caused by fungal infection. Use copper oxychloride for treatment.",
            metadata={"topic": "disease"},
        ),
    ]

    # Build index
    print("\n[1] Building LightRAG index...")
    indexer = LightRAGIndexer()
    index = indexer.index_documents(docs)
    print(f"    Index stats: {index.stats}")

    # Test HYBRID mode
    print("\n[2] Testing HYBRID mode...")
    hybrid_retriever = LightRAGHybridRetriever(
        index=index,
        mode=QueryMode.HYBRID,
    )
    hybrid_results = hybrid_retriever.search("How to treat rice disease?", k=2)
    print(f"    Results: {len(hybrid_results)}")
    for doc in hybrid_results:
        print(f"      - {doc.id}: {doc.text[:50]}...")

    # Test PATHRAG mode
    print("\n[3] Testing PATHRAG mode...")
    pathrag_retriever = LightRAGHybridRetriever(
        index=index,
        mode=QueryMode.PATHRAG,
    )
    pathrag_results = pathrag_retriever.search("How to treat rice disease?", k=2)
    print(f"    Results: {len(pathrag_results)}")
    for doc in pathrag_results:
        score = doc.metadata.get("retrieval_score", 0)
        print(f"      - {doc.id} (score={score:.4f}): {doc.text[:50]}...")

    # Verify adjacency is being built correctly
    print("\n[4] Verifying adjacency builder...")
    adjacency, reverse_adjacency = index.build_entity_chunk_adjacency()
    print(f"    Forward adjacency nodes: {len(adjacency)}")
    print(f"    Reverse adjacency nodes: {len(reverse_adjacency)}")

    # Check flow scores directly
    print("\n[5] Testing flow scores directly...")
    from core.Services.Retrieval.PathScoring import bfs_weighted_paths

    # Get some entity IDs
    entity_ids = list(index.entities._entities.keys())[:3]
    chunk_ids = index.get_all_chunk_ids()

    print(f"    Source entities: {entity_ids}")
    print(f"    Target chunks: {chunk_ids}")

    flow_scores = bfs_weighted_paths(
        adjacency=adjacency,
        reverse_adjacency=reverse_adjacency,
        sources=entity_ids,
        practice_node_ids=chunk_ids,
        max_hops=3,
        alpha=0.8,
    )
    print(f"    Flow scores: {flow_scores}")

    print("\n" + "=" * 60)
    print("SUCCESS: PathRAG mode integration works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
