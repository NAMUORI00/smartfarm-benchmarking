"""Build LLM-based knowledge graph from AgriQA corpus using CausalExtractor."""

from __future__ import annotations

import asyncio
from pathlib import Path
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Set LLMLITE_HOST to external Docker port
os.environ["LLMLITE_HOST"] = "http://localhost:45857"

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas.BaseSchemas import SourceDoc
from core.Services.Ingest.CausalExtractor import CausalExtractor, EntityNormalizer
from core.Services.Ingest.LLMGraphBuilder import build_graph_from_extractions


async def main():
    """Build LLM-based graph from AgriQA corpus."""

    # Configuration
    corpus_path = PROJECT_ROOT / "benchmarking" / "data" / "agriqa" / "corpus.jsonl"
    output_path = PROJECT_ROOT / "output" / "benchmarking" / "smartfarm_graph_llm.json"
    max_docs = 10000  # Process all documents

    print(f"Loading corpus from: {corpus_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            data = json.loads(line)
            doc = SourceDoc(
                id=data["_id"],  # AgriQA uses _id field
                text=data["text"],
                metadata={"title": data.get("title", "")},
            )
            docs.append(doc)

    print(f"Loaded {len(docs)} documents")

    # Initialize extractor
    print("Initializing CausalExtractor...")
    extractor = CausalExtractor(
        batch_size=5,
        min_confidence=0.7,
    )

    # Extract entities and relations
    print("Extracting entities and relations from documents...")
    print(f"LLM Server: {os.environ['LLMLITE_HOST']}")
    results = await extractor.extract(docs)

    print(f"\nExtraction complete:")
    print(f"  - Processed {len(results)} documents")
    total_entities = sum(len(r.entities) for r in results)
    total_relations = sum(len(r.relations) for r in results)
    print(f"  - Extracted {total_entities} entities")
    print(f"  - Extracted {total_relations} relations")

    # Normalize entities
    print("\nNormalizing entities...")
    normalizer = EntityNormalizer()
    normalized_results = normalizer.normalize(results)

    # Build graph
    print("Building knowledge graph...")
    graph = build_graph_from_extractions(
        results=normalized_results,
        docs=docs,
        graph=None,
        generate_embeddings=False,  # Skip embeddings for speed
    )

    print(f"\nGraph statistics:")
    print(f"  - Nodes: {len(graph.nodes)}")
    print(f"  - Edges: {len(graph.edges)}")

    # Count by node type
    node_types = {}
    for node in graph.nodes.values():
        node_types[node.type] = node_types.get(node.type, 0) + 1
    print(f"  - Node types: {node_types}")

    # Count by edge type
    edge_types = {}
    for edge in graph.edges:
        edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
    print(f"  - Edge types: {edge_types}")

    # Save graph
    print(f"\nSaving graph to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_data = {
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "name": node.name,
                "description": node.description,
                "metadata": node.metadata,
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "weight": edge.weight,
                "metadata": edge.metadata,
            }
            for edge in graph.edges
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"Graph saved successfully!")
    print(f"\nYou can now use this graph with PATHRAG_LT_SEED_MODE=vector")


if __name__ == "__main__":
    asyncio.run(main())
