"""Compare all 4 seed matching strategies for PathRAGLtRetriever.

Tests each strategy (ONTOLOGY, KEYWORD, METADATA, ALL_CONCEPTS) with various query types
and measures seed count, result count, and coverage.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Graph import SmartFarmGraph, GraphNode, GraphEdge
from core.Services.Retrieval.PathRAG import PathRAGLtRetriever, SeedMatchMode


def create_test_graph() -> SmartFarmGraph:
    """Create a test graph with known structure for seed strategy testing."""
    graph = SmartFarmGraph()

    # === Concept nodes (potential seeds) ===
    # Crops
    graph.add_node(GraphNode(id="crop:wasabi", type="crop", name="와사비", description="Wasabi crop"))
    graph.add_node(GraphNode(id="crop:tomato", type="crop", name="토마토", description="Tomato crop"))

    # Diseases
    graph.add_node(GraphNode(id="disease:powdery_mildew", type="disease", name="흰가루병", description="Powdery mildew disease"))
    graph.add_node(GraphNode(id="disease:root_rot", type="disease", name="뿌리썩음병", description="Root rot disease"))

    # Environmental factors
    graph.add_node(GraphNode(id="env:temperature", type="env", name="온도", description="Temperature control"))
    graph.add_node(GraphNode(id="env:humidity", type="env", name="습도", description="Humidity control"))
    graph.add_node(GraphNode(id="env:ec", type="env", name="EC", description="Electrical conductivity"))

    # Growth stages
    graph.add_node(GraphNode(id="stage:growth", type="stage", name="생육기", description="Growth stage"))
    graph.add_node(GraphNode(id="stage:seedling", type="stage", name="육묘기", description="Seedling stage"))

    # === Practice nodes (targets) ===
    graph.add_node(GraphNode(
        id="practice:p1", type="practice", name="온도 관리",
        description="와사비 재배 시 적정 온도는 13-17도입니다.",
        metadata={"작물": "와사비", "주제": "온도"}
    ))
    graph.add_node(GraphNode(
        id="practice:p2", type="practice", name="병해 방제",
        description="흰가루병 발생 시 환기를 강화하고 살균제를 살포합니다.",
        metadata={"작물": "와사비", "주제": "병해"}
    ))
    graph.add_node(GraphNode(
        id="practice:p3", type="practice", name="양액 관리",
        description="EC 1.5-2.0 범위를 유지하고 pH는 6.0 전후로 조절합니다.",
        metadata={"작물": "와사비", "주제": "양액"}
    ))
    graph.add_node(GraphNode(
        id="practice:p4", type="practice", name="습도 관리",
        description="습도 80% 이상 유지, 환기 필요",
        metadata={"작물": "와사비", "주제": "환경"}
    ))
    graph.add_node(GraphNode(
        id="practice:p5", type="practice", name="육묘 관리",
        description="육묘기 온도는 18-20도 유지",
        metadata={"작물": "와사비", "주제": "육묘"}
    ))
    graph.add_node(GraphNode(
        id="practice:p6", type="practice", name="생육 환경",
        description="생육기 환경 관리는 온도, 습도, EC를 종합적으로 고려합니다.",
        metadata={"작물": "와사비", "주제": "생육"}
    ))

    # === Edges ===
    # 1-hop: crop -> practices
    graph.add_edge(GraphEdge(source="crop:wasabi", target="practice:p1", type="recommended_for", weight=1.0))
    graph.add_edge(GraphEdge(source="crop:wasabi", target="practice:p3", type="recommended_for", weight=1.0))

    # 1-hop: disease -> practice
    graph.add_edge(GraphEdge(source="disease:powdery_mildew", target="practice:p2", type="solved_by", weight=1.0))

    # 2-hop: env -> stage -> practice
    graph.add_edge(GraphEdge(source="env:temperature", target="stage:growth", type="associated_with", weight=1.0))
    graph.add_edge(GraphEdge(source="stage:growth", target="practice:p1", type="associated_with", weight=1.0))
    graph.add_edge(GraphEdge(source="stage:growth", target="practice:p6", type="associated_with", weight=1.0))

    # 2-hop: env -> practice paths
    graph.add_edge(GraphEdge(source="env:humidity", target="practice:p4", type="controls", weight=1.0))
    graph.add_edge(GraphEdge(source="env:ec", target="practice:p3", type="controls", weight=1.0))

    # 2-hop: stage paths
    graph.add_edge(GraphEdge(source="stage:seedling", target="practice:p5", type="associated_with", weight=1.0))

    return graph


def test_seed_strategy(graph: SmartFarmGraph, query: str, mode: str) -> tuple[int, int, set]:
    """Test a single seed matching strategy.

    Returns:
        (seed_count, result_count, result_ids_set)
    """
    retriever = PathRAGLtRetriever(
        graph=graph,
        max_hops=3,
        threshold=0.05,  # Low threshold for small test graph
        alpha=0.8,
        max_results=10,
        seed_match_mode=mode,
    )

    # Get seeds (use internal method for inspection)
    seeds = retriever._match_seeds(query)
    seed_count = len(seeds)

    # Get results
    results = retriever.search(query, k=10)
    result_count = len(results)
    result_ids = {r.id for r in results}

    return seed_count, result_count, result_ids


def calculate_coverage(result_ids: set, all_practice_ids: set) -> float:
    """Calculate coverage percentage."""
    if not all_practice_ids:
        return 0.0
    return (len(result_ids) / len(all_practice_ids)) * 100


def run_comparison():
    """Run comprehensive seed strategy comparison."""
    graph = create_test_graph()

    # Get all practice IDs for coverage calculation
    all_practice_ids = {nid for nid, n in graph.nodes.items() if n.type == "practice"}
    total_practices = len(all_practice_ids)

    # Test queries with different characteristics
    test_queries = [
        ("와사비 재배 시 적정 온도는?", "Korean - Specific topic"),
        ("흰가루병 방제 방법", "Korean - Disease keyword"),
        ("wasabi temperature control", "English - Keyword"),
        ("생육 환경 관리", "Korean - Vague/broad"),
        ("EC 농도 조절", "Korean - Technical term"),
    ]

    modes = [
        ("ontology", "ONTOLOGY"),
        ("keyword", "KEYWORD"),
        ("metadata", "METADATA"),
        ("all", "ALL_CONCEPTS"),
    ]

    print("=" * 80)
    print("SEED MATCHING STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Test graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Practice nodes (targets): {total_practices}")
    print()

    for query, desc in test_queries:
        print(f"Query: \"{query}\"")
        print(f"Type: {desc}")
        print("-" * 80)
        print(f"{'Mode':<15} | {'Seeds':<6} | {'Results':<7} | {'Coverage':<10} | Practice IDs")
        print("-" * 80)

        for mode_val, mode_name in modes:
            seed_count, result_count, result_ids = test_seed_strategy(graph, query, mode_val)
            coverage = calculate_coverage(result_ids, all_practice_ids)

            # Truncate IDs for display
            id_display = ", ".join(sorted(result_ids)[:3])
            if len(result_ids) > 3:
                id_display += f" ... (+{len(result_ids)-3})"

            print(f"{mode_name:<15} | {seed_count:^6} | {result_count:^7} | {coverage:>6.1f}%   | {id_display}")

        print()

    # Summary insights
    print("=" * 80)
    print("SUMMARY INSIGHTS")
    print("=" * 80)
    print()
    print("ONTOLOGY Mode:")
    print("  - Strictest matching (uses predefined ontology)")
    print("  - Lowest coverage, high precision")
    print("  - May miss results if ontology is incomplete")
    print()
    print("KEYWORD Mode:")
    print("  - Flexible keyword + partial matching")
    print("  - Balanced precision/recall")
    print("  - RECOMMENDED as default strategy")
    print()
    print("METADATA Mode:")
    print("  - Uses predefined keyword mappings")
    print("  - Good for structured queries (crop names, topics)")
    print("  - Limited to known entities")
    print()
    print("ALL_CONCEPTS Mode:")
    print("  - Uses all concept nodes as seeds")
    print("  - Maximum coverage, may reduce precision")
    print("  - Best for broad exploration queries")
    print()

    # Detailed seed inspection for one query
    print("=" * 80)
    print("DETAILED SEED INSPECTION")
    print("=" * 80)
    query = "와사비 재배 시 적정 온도는?"
    print(f"Query: \"{query}\"")
    print()

    for mode_val, mode_name in modes:
        retriever = PathRAGLtRetriever(graph, seed_match_mode=mode_val)
        seeds = retriever._match_seeds(query)

        print(f"{mode_name} Seeds ({len(seeds)}):")
        for seed in seeds[:5]:  # Show first 5
            print(f"  - {seed.id} (type={seed.type}, name={seed.name})")
        if len(seeds) > 5:
            print(f"  ... and {len(seeds)-5} more")
        print()


if __name__ == "__main__":
    run_comparison()
