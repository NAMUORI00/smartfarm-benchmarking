#!/usr/bin/env python3
"""Unit tests for CROP corpus builder."""

import json
import tempfile
from pathlib import Path

from benchmarking.data.crop_corpus_builder import (
    compute_similarity,
    deduplicate_contexts,
    extract_corpus_and_mappings,
    generate_doc_id,
    load_qa_pairs,
)


def test_generate_doc_id():
    """Test deterministic document ID generation."""
    text = "Sample context text"
    doc_id1 = generate_doc_id(text)
    doc_id2 = generate_doc_id(text)

    assert doc_id1 == doc_id2
    assert len(doc_id1) == 16
    assert doc_id1.isalnum()


def test_compute_similarity():
    """Test Jaccard similarity computation."""
    # Identical texts
    assert compute_similarity("hello world", "hello world") == 1.0

    # No overlap
    assert compute_similarity("hello world", "foo bar") == 0.0

    # Partial overlap
    sim = compute_similarity("hello world", "hello foo")
    assert 0 < sim < 1

    # Empty texts
    assert compute_similarity("", "") == 0.0
    assert compute_similarity("hello", "") == 0.0


def test_deduplicate_contexts():
    """Test context deduplication."""
    contexts = [
        ("id1", "This is a test context"),
        ("id2", "This is a test context"),  # Duplicate
        ("id3", "Completely different text"),
        ("id4", "This is a test context with more"),  # Similar but different
    ]

    # High threshold (keep more docs)
    unique = deduplicate_contexts(contexts, threshold=0.95)
    assert len(unique) >= 2

    # Low threshold (more aggressive dedup)
    unique = deduplicate_contexts(contexts, threshold=0.5)
    assert len(unique) <= 3


def test_load_qa_pairs():
    """Test QA pair loading from JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test QA file
        qa_file = tmpdir / "wasabi_qa_test.jsonl"
        with open(qa_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": "qa_001",
                        "question": "What is wasabi?",
                        "answer": "A plant...",
                        "context": "Wasabi is a plant...",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "id": "qa_002",
                        "question": "How to grow?",
                        "answer": "In shade...",
                        "context": "Grow in shaded areas...",
                    }
                )
                + "\n"
            )

        # Create non-QA file (should be ignored)
        other_file = tmpdir / "wasabi_other.jsonl"
        with open(other_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "other", "text": "Some text"}) + "\n")

        # Load QA pairs
        qa_pairs = load_qa_pairs(tmpdir)

        assert len(qa_pairs) == 2
        assert qa_pairs[0]["id"] == "qa_001"
        assert qa_pairs[1]["id"] == "qa_002"


def test_load_qa_pairs_with_limit():
    """Test QA pair loading with limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        qa_file = tmpdir / "wasabi_qa_test.jsonl"
        with open(qa_file, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(
                    json.dumps(
                        {
                            "id": f"qa_{i:03d}",
                            "question": f"Question {i}",
                            "answer": f"Answer {i}",
                            "context": f"Context {i}",
                        }
                    )
                    + "\n"
                )

        qa_pairs = load_qa_pairs(tmpdir, limit=5)
        assert len(qa_pairs) == 5


def test_extract_corpus_and_mappings():
    """Test corpus extraction and query-document mapping."""
    qa_pairs = [
        {
            "id": "qa_001",
            "question": "What is wasabi?",
            "answer": "A plant",
            "context": "Wasabi is a plant in the Brassicaceae family.",
        },
        {
            "id": "qa_002",
            "question": "Where does it grow?",
            "answer": "In Japan",
            "context": "Wasabi grows in mountain streams in Japan.",
        },
        {
            "id": "qa_003",
            "question": "What family?",
            "answer": "Brassicaceae",
            "context": "Wasabi is a plant in the Brassicaceae family.",  # Duplicate context
        },
    ]

    corpus, queries, qrels = extract_corpus_and_mappings(qa_pairs, dedup_threshold=0.95)

    # Check corpus
    assert len(corpus) == 2  # Duplicate removed
    assert all(len(text) > 0 for text in corpus.values())

    # Check queries
    assert len(queries) == 3
    assert queries["qa_001"] == "What is wasabi?"
    assert queries["qa_002"] == "Where does it grow?"
    assert queries["qa_003"] == "What family?"

    # Check qrels
    assert len(qrels) == 3
    assert all(len(docs) == 1 for docs in qrels.values())
    assert qrels["qa_001"][0] == qrels["qa_003"][0]  # Same context, same doc


def test_extract_corpus_empty_contexts():
    """Test handling of empty contexts."""
    qa_pairs = [
        {
            "id": "qa_001",
            "question": "Question 1",
            "answer": "Answer 1",
            "context": "",  # Empty
        },
        {
            "id": "qa_002",
            "question": "Question 2",
            "answer": "Answer 2",
            "context": "Valid context",
        },
    ]

    corpus, queries, qrels = extract_corpus_and_mappings(qa_pairs)

    assert len(corpus) == 1  # Only valid context
    assert len(queries) == 2  # Both queries kept
    assert len(qrels) == 1  # Only qa_002 has valid mapping


def test_beir_format_compliance():
    """Test that generated files comply with BEIR format."""
    qa_pairs = [
        {
            "id": "qa_001",
            "question": "What is wasabi?",
            "answer": "A plant",
            "context": "Wasabi is a plant in the Brassicaceae family.",
        }
    ]

    corpus, queries, qrels = extract_corpus_and_mappings(qa_pairs)

    # Corpus format: {"_id": str, "title": str, "text": str}
    for doc_id, text in corpus.items():
        assert isinstance(doc_id, str)
        assert len(doc_id) == 16  # Hash length
        assert isinstance(text, str)
        assert len(text) > 0

    # Queries format: {"_id": str, "text": str}
    for query_id, text in queries.items():
        assert isinstance(query_id, str)
        assert isinstance(text, str)
        assert len(text) > 0

    # Qrels format: query_id -> list of doc_ids with score 1
    for query_id, doc_ids in qrels.items():
        assert isinstance(query_id, str)
        assert isinstance(doc_ids, list)
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        assert all(doc_id in corpus for doc_id in doc_ids)


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_generate_doc_id,
        test_compute_similarity,
        test_deduplicate_contexts,
        test_load_qa_pairs,
        test_load_qa_pairs_with_limit,
        test_extract_corpus_and_mappings,
        test_extract_corpus_empty_contexts,
        test_beir_format_compliance,
    ]

    print("Running CROP corpus builder tests...\n")
    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"[PASS] {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_func.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    exit(0 if failed == 0 else 1)
