#!/usr/bin/env python3
"""Validation test for CROP LightRAG evaluation script.

Quick smoke test to ensure the evaluation pipeline works correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmarking.experiments.crop_lightrag_eval import (
    load_crop_corpus,
    load_crop_queries,
    load_crop_qrels,
    build_lightrag,
    evaluate_lightrag,
    get_graph_stats,
)


def test_data_loading():
    """Test CROP data loading functions."""
    print("Testing data loading...")

    data_dir = Path("data/crop")

    # Test corpus loading
    docs = load_crop_corpus(data_dir, doc_limit=10)
    assert len(docs) == 10, f"Expected 10 docs, got {len(docs)}"
    assert all(doc.id for doc in docs), "All docs should have IDs"
    assert all(doc.text for doc in docs), "All docs should have text"
    print(f"✓ Loaded {len(docs)} documents")

    # Test queries loading
    queries = load_crop_queries(data_dir)
    assert len(queries) > 0, "Should load at least 1 query"
    assert all(isinstance(qid, str) for qid in queries.keys()), "Query IDs should be strings"
    assert all(isinstance(text, str) for text in queries.values()), "Query texts should be strings"
    print(f"✓ Loaded {len(queries)} queries")

    # Test qrels loading
    qrels = load_crop_qrels(data_dir)
    assert len(qrels) > 0, "Should load at least 1 qrel"
    assert all(isinstance(qid, str) for qid in qrels.keys()), "Qrel query IDs should be strings"
    print(f"✓ Loaded qrels for {len(qrels)} queries")

    print("Data loading tests passed!\n")
    return docs, queries, qrels


def test_lightrag_build(docs):
    """Test LightRAG graph construction."""
    print("Testing LightRAG build...")

    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir) / "lightrag_test"

        # Build LightRAG with small corpus
        lightrag = build_lightrag(docs[:5], working_dir)

        assert lightrag is not None, "LightRAG should be built"
        assert working_dir.exists(), "Working directory should exist"
        assert len(list(working_dir.iterdir())) > 0, "Working directory should have files"
        print(f"✓ LightRAG built successfully in {working_dir}")

        # Test graph stats
        stats = get_graph_stats(working_dir)
        print(f"✓ Graph stats: {stats}")

    print("LightRAG build tests passed!\n")


def test_evaluation(docs, queries, qrels):
    """Test evaluation metrics computation."""
    print("Testing evaluation metrics...")

    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir) / "lightrag_eval_test"

        # Build LightRAG with small corpus
        lightrag = build_lightrag(docs[:10], working_dir)

        # Run evaluation on limited queries
        eval_results = evaluate_lightrag(
            lightrag=lightrag,
            queries=queries,
            qrels=qrels,
            top_k=4,
            max_queries=5,  # Only evaluate 5 queries for speed
        )

        # Validate results structure
        assert "metrics" in eval_results, "Should have metrics"
        assert "latency" in eval_results, "Should have latency stats"
        assert "n_queries_evaluated" in eval_results, "Should have query count"

        metrics = eval_results["metrics"]
        assert "mrr" in metrics, "Should have MRR"
        assert "ndcg@4" in metrics, "Should have NDCG@4"
        assert "precision@4" in metrics, "Should have Precision@4"
        assert "recall@4" in metrics, "Should have Recall@4"
        assert "hit_rate@4" in metrics, "Should have Hit Rate@4"

        # Validate metric ranges
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} should be in [0, 1], got {value}"

        print(f"✓ Evaluation completed on {eval_results['n_queries_evaluated']} queries")
        print("✓ Metrics:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

    print("Evaluation tests passed!\n")


def test_json_output():
    """Test JSON output format."""
    print("Testing JSON output format...")

    # Create sample output
    output_data = {
        "dataset": "crop",
        "method": "lightrag",
        "n_docs": 10,
        "n_queries": 220,
        "n_queries_evaluated": 5,
        "top_k": 4,
        "metrics": {
            "mrr": 0.5,
            "ndcg@4": 0.6,
            "precision@4": 0.4,
            "recall@4": 0.3,
            "hit_rate@4": 0.8,
        },
        "latency": {
            "mean_ms": 100.0,
            "p50_ms": 80.0,
            "p95_ms": 200.0,
        },
        "graph_stats": {
            "n_nodes": 50,
            "n_edges": 120,
        },
        "config": {
            "query_mode": "hybrid",
            "data_dir": "data/crop",
            "working_dir": "output/test",
        },
    }

    # Test JSON serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        temp_path = f.name

    # Test JSON loading
    with open(temp_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)

    assert loaded_data == output_data, "JSON round-trip should preserve data"
    print("✓ JSON output format is valid")

    # Cleanup
    Path(temp_path).unlink()

    print("JSON output tests passed!\n")


def main():
    """Run all validation tests."""
    print("="*60)
    print("CROP LightRAG Evaluation - Validation Tests")
    print("="*60)
    print()

    try:
        # Test data loading
        docs, queries, qrels = test_data_loading()

        # Test LightRAG build
        test_lightrag_build(docs)

        # Test evaluation
        test_evaluation(docs, queries, qrels)

        # Test JSON output
        test_json_output()

        print("="*60)
        print("All validation tests passed! ✓")
        print("="*60)
        print()
        print("The script is ready to run full evaluation:")
        print("  python -m benchmarking.experiments.crop_lightrag_eval")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
