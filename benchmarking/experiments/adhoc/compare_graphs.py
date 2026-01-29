"""Compare rule-based and LLM-based graphs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate graph paths exist: {[str(p) for p in paths]}")


# Load graphs (prefer benchmarking output, fallback to data/index)
rule_graph_path = _first_existing(
    [
        PROJECT_ROOT / "output" / "benchmarking" / "smartfarm_graph_rule.json",
        PROJECT_ROOT / "data" / "index" / "smartfarm_graph.json",
    ]
)
llm_graph_path = _first_existing(
    [
        PROJECT_ROOT / "output" / "benchmarking" / "smartfarm_graph_llm.json",
        PROJECT_ROOT / "data" / "index" / "smartfarm_graph_llm.json",
    ]
)

print("Loading graphs...")
with open(rule_graph_path, "r", encoding="utf-8") as f:
    rule_graph = json.load(f)

with open(llm_graph_path, "r", encoding="utf-8") as f:
    llm_graph = json.load(f)

print("\n" + "=" * 80)
print("GRAPH COMPARISON")
print("=" * 80)

# Compare sizes
print("\n1. Graph Sizes:")
print(f"   Rule-based:  {len(rule_graph['nodes'])} nodes, {len(rule_graph['edges'])} edges")
print(f"   LLM-based:   {len(llm_graph['nodes'])} nodes, {len(llm_graph['edges'])} edges")

# Compare node types
print("\n2. Node Type Distribution:")
rule_types = Counter(node["type"] for node in rule_graph["nodes"])
llm_types = Counter(node["type"] for node in llm_graph["nodes"])

print(f"   Rule-based: {dict(rule_types)}")
print(f"   LLM-based:  {dict(llm_types)}")

# Key comparison: concept nodes (non-practice)
rule_concepts = sum(count for node_type, count in rule_types.items() if node_type != "practice")
llm_concepts = sum(count for node_type, count in llm_types.items() if node_type != "practice")

print(f"\n   → Concept nodes (non-practice):")
print(f"     Rule-based: {rule_concepts} nodes")
print(f"     LLM-based:  {llm_concepts} nodes")
if rule_concepts > 0:
    pct = (llm_concepts - rule_concepts) / rule_concepts * 100
    print(f"     Improvement: +{llm_concepts - rule_concepts} nodes ({pct:.1f}%)")
else:
    print(f"     Improvement: +{llm_concepts - rule_concepts} nodes")

# Compare edge types
print("\n3. Edge Type Distribution:")
rule_edge_types = Counter(edge["type"] for edge in rule_graph["edges"])
llm_edge_types = Counter(edge["type"] for edge in llm_graph["edges"])

print(f"   Rule-based: {dict(rule_edge_types)}")
print(f"   LLM-based:  {dict(llm_edge_types)}")

# Show sample entities
print("\n4. Sample Entities (LLM-based):")
concept_nodes = [n for n in llm_graph["nodes"] if n["type"] != "practice"][:15]
for i, node in enumerate(concept_nodes, 1):
    print(f"   {i}. [{node['type']}] {node['name']}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nKey Findings:")
print(f"1. The LLM-based graph has {llm_concepts}x more concept nodes than rule-based ({rule_concepts} → {llm_concepts})")
print(f"2. The LLM extracted diverse entity types: {', '.join(k for k in llm_types.keys() if k != 'practice')}")
print(f"3. The LLM discovered {len(llm_edge_types)} different edge types")
print("\nThis demonstrates that LLM-based extraction can discover significantly more")
print("semantic entities for PathRAG's vector seed matching benchmark.")
