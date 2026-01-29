"""Direct comparison of PathRAGRetriever vs PathRAGLtRetriever without ontology."""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Graph import SmartFarmGraph, GraphNode, GraphEdge
from core.Services.Retrieval.PathRAG import PathRAGRetriever, PathRAGLtRetriever


def create_test_graph() -> SmartFarmGraph:
    """Create a test graph with known structure."""
    graph = SmartFarmGraph()

    # Add concept nodes (seeds)
    graph.add_node(GraphNode(id="crop:wasabi", type="crop", name="와사비", description=None))
    graph.add_node(GraphNode(id="disease:powdery_mildew", type="disease", name="흰가루병", description=None))
    graph.add_node(GraphNode(id="env:temperature", type="env", name="온도", description=None))

    # Add practice nodes (targets)
    graph.add_node(GraphNode(
        id="practice:p1", type="practice", name="온도 관리",
        description="와사비 재배 시 적정 온도는 13-17도입니다.",
        metadata={"작물": "와사비"}
    ))
    graph.add_node(GraphNode(
        id="practice:p2", type="practice", name="병해 방제",
        description="흰가루병 발생 시 환기를 강화하고 살균제를 살포합니다.",
        metadata={"작물": "와사비"}
    ))
    graph.add_node(GraphNode(
        id="practice:p3", type="practice", name="양액 관리",
        description="EC 1.5-2.0 범위를 유지하고 pH는 6.0 전후로 조절합니다.",
        metadata={"작물": "와사비"}
    ))
    graph.add_node(GraphNode(
        id="practice:p4", type="practice", name="습도 관리",
        description="습도 80% 이상 유지, 환기 필요",
        metadata={"작물": "와사비"}
    ))

    # Add intermediate node
    graph.add_node(GraphNode(id="stage:growth", type="stage", name="생육기", description=None))

    # Add edges: crop -> practice (1-hop)
    graph.add_edge(GraphEdge(source="crop:wasabi", target="practice:p1", type="recommended_for", weight=1.0))
    graph.add_edge(GraphEdge(source="crop:wasabi", target="practice:p3", type="recommended_for", weight=1.0))

    # Add edges: disease -> practice (1-hop)
    graph.add_edge(GraphEdge(source="disease:powdery_mildew", target="practice:p2", type="solved_by", weight=1.0))

    # Add edges: env -> stage -> practice (2-hop path)
    graph.add_edge(GraphEdge(source="env:temperature", target="stage:growth", type="associated_with", weight=1.0))
    graph.add_edge(GraphEdge(source="stage:growth", target="practice:p1", type="associated_with", weight=1.0))
    graph.add_edge(GraphEdge(source="stage:growth", target="practice:p4", type="associated_with", weight=1.0))

    return graph


def test_direct_search():
    """Test both retrievers with direct seed nodes."""
    graph = create_test_graph()

    old_retriever = PathRAGRetriever(graph)
    new_retriever = PathRAGLtRetriever(graph, threshold=0.1)

    print("=" * 70)
    print("PathRAG Direct Comparison (No Ontology)")
    print("=" * 70)

    # Test 1: 1-hop path from crop node
    print("\n[Test 1] Seed: crop:wasabi (1-hop to practices)")

    # Manually call internal methods to bypass ontology
    old_results = []
    new_results = []

    # For old PathRAG, simulate search from matched node
    start_node = graph.nodes["crop:wasabi"]
    queue = [(start_node.id, 0)]
    visited = {start_node.id}
    seen_practices = set()

    while queue:
        nid, depth = queue.pop(0)
        node = graph.nodes.get(nid)
        if node and node.type == "practice" and node.id not in seen_practices:
            seen_practices.add(node.id)
            old_results.append((node.id, node.description))
        if depth < 2:
            for nb in graph.get_neighbors(nid):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))

    print(f"  Old PathRAG: {len(old_results)} results")
    for rid, desc in old_results[:5]:
        print(f"    - {rid}: {desc[:40]}...")

    # For new PathRAG-lt, use the scoring algorithm directly
    from core.Services.Retrieval.PathScoring import bfs_weighted_paths, balanced_path_selection

    practice_ids = {nid for nid, n in graph.nodes.items() if n.type == "practice"}
    scores = bfs_weighted_paths(
        adjacency=graph.adjacency,
        reverse_adjacency=graph.reverse_adjacency,
        sources=["crop:wasabi"],
        practice_node_ids=practice_ids,
        max_hops=3,
        threshold=0.1,
        alpha=0.8,
        use_soft_normalization=True,
        prune_during_exploration=False,
    )

    selected = balanced_path_selection(scores, max_results=10)
    print(f"  New PathRAG-lt: {len(selected)} results")
    for nid, score in selected[:5]:
        node = graph.nodes.get(nid)
        desc = node.description if node else "N/A"
        print(f"    - {nid} (score={score:.4f}): {desc[:40]}...")

    # Test 2: 2-hop path from env node
    print("\n[Test 2] Seed: env:temperature (2-hop path via stage)")

    # Old PathRAG
    old_results_2 = []
    start_node = graph.nodes["env:temperature"]
    queue = [(start_node.id, 0)]
    visited = {start_node.id}
    seen_practices = set()

    while queue:
        nid, depth = queue.pop(0)
        node = graph.nodes.get(nid)
        if node and node.type == "practice" and node.id not in seen_practices:
            seen_practices.add(node.id)
            old_results_2.append((node.id, node.description))
        if depth < 2:
            for nb in graph.get_neighbors(nid):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))

    print(f"  Old PathRAG: {len(old_results_2)} results")
    for rid, desc in old_results_2[:5]:
        print(f"    - {rid}: {desc[:40]}...")

    # New PathRAG-lt
    scores_2 = bfs_weighted_paths(
        adjacency=graph.adjacency,
        reverse_adjacency=graph.reverse_adjacency,
        sources=["env:temperature"],
        practice_node_ids=practice_ids,
        max_hops=3,
        threshold=0.1,
        alpha=0.8,
        use_soft_normalization=True,
        prune_during_exploration=False,
    )

    selected_2 = balanced_path_selection(scores_2, max_results=10)
    print(f"  New PathRAG-lt: {len(selected_2)} results")
    for nid, score in selected_2[:5]:
        node = graph.nodes.get(nid)
        desc = node.description if node else "N/A"
        print(f"    - {nid} (score={score:.4f}): {desc[:40]}...")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Test 1 (1-hop): Old={len(old_results)}, New={len(selected)} results")
    print(f"Test 2 (2-hop): Old={len(old_results_2)}, New={len(selected_2)} results")

    if selected:
        print(f"\nNew PathRAG-lt advantage: Provides confidence scores for ranking!")
        print(f"  Top score: {selected[0][1]:.4f}")
        if len(selected) > 1:
            print(f"  2nd score: {selected[1][1]:.4f}")


if __name__ == "__main__":
    test_direct_search()
