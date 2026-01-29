#!/usr/bin/env python3
"""CROP-dataset corpus builder for BEIR-compatible format.

This module extracts unique context passages from QA pairs and generates
BEIR-compatible corpus, queries, and qrels files for retrieval evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def generate_doc_id(text: str) -> str:
    """Generate deterministic document ID from content hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute_similarity(text1: str, text2: str) -> float:
    """Compute simple token-based Jaccard similarity."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def deduplicate_contexts(
    contexts: List[Tuple[str, str]],
    threshold: float = 0.95,
) -> Dict[str, str]:
    """Deduplicate similar contexts using similarity threshold.

    Args:
        contexts: List of (doc_id, text) tuples
        threshold: Similarity threshold for deduplication

    Returns:
        Dict mapping doc_id to unique context text
    """
    unique_docs: Dict[str, str] = {}
    seen_texts: List[str] = []

    for doc_id, text in contexts:
        is_duplicate = False
        for seen_text in seen_texts:
            if compute_similarity(text, seen_text) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs[doc_id] = text
            seen_texts.append(text)

    return unique_docs


def load_qa_pairs(input_dir: Path, limit: int | None = None) -> List[Dict]:
    """Load QA pairs from JSONL files in input directory.

    Only loads files that contain QA pairs (have 'question', 'answer', 'context' fields).
    """
    qa_pairs: List[Dict] = []

    for jsonl_file in input_dir.glob("*qa*.jsonl"):
        print(f"[CROP] Reading {jsonl_file.name}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Validate QA structure
                if "question" in data and "answer" in data and "context" in data:
                    qa_pairs.append(data)

                if limit is not None and len(qa_pairs) >= limit:
                    return qa_pairs

    return qa_pairs


def extract_corpus_and_mappings(
    qa_pairs: List[Dict],
    dedup_threshold: float = 0.95,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """Extract unique corpus documents and create query-document mappings.

    Args:
        qa_pairs: List of QA pair dictionaries
        dedup_threshold: Similarity threshold for deduplication

    Returns:
        Tuple of (corpus_dict, queries_dict, qrels_dict)
        - corpus_dict: doc_id -> context text
        - queries_dict: query_id -> question text
        - qrels_dict: query_id -> list of relevant doc_ids
    """
    # Collect all contexts with their source doc IDs
    contexts: List[Tuple[str, str]] = []
    query_to_context: Dict[str, str] = {}

    for qa in qa_pairs:
        context = qa.get("context", "").strip()
        if not context:
            continue

        doc_id = generate_doc_id(context)
        contexts.append((doc_id, context))

        # Store mapping from query to context for qrels
        query_id = qa.get("id", "")
        if query_id:
            query_to_context[query_id] = doc_id

    # Deduplicate contexts
    corpus = deduplicate_contexts(contexts, threshold=dedup_threshold)

    # Build queries dict
    queries: Dict[str, str] = {}
    for qa in qa_pairs:
        query_id = qa.get("id", "")
        question = qa.get("question", "").strip()
        if query_id and question:
            queries[query_id] = question

    # Build qrels: map each query to its relevant documents
    qrels: Dict[str, List[str]] = defaultdict(list)
    for query_id, doc_id in query_to_context.items():
        if doc_id in corpus and query_id in queries:
            qrels[query_id].append(doc_id)

    return corpus, queries, dict(qrels)


def write_corpus(corpus: Dict[str, str], output_path: Path) -> None:
    """Write corpus to BEIR-compatible JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            doc = {
                "_id": doc_id,
                "title": "",
                "text": text,
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def write_queries(queries: Dict[str, str], output_path: Path) -> None:
    """Write queries to BEIR-compatible JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for query_id, text in queries.items():
            query = {
                "_id": query_id,
                "text": text,
            }
            f.write(json.dumps(query, ensure_ascii=False) + "\n")


def write_qrels(qrels: Dict[str, List[str]], output_path: Path) -> None:
    """Write qrels to BEIR-compatible TSV format.

    Format: query-id \t corpus-id \t score
    Binary relevance: 1 if document is relevant to query
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("query-id\tcorpus-id\tscore\n")

        # Write relevance judgments
        for query_id, doc_ids in sorted(qrels.items()):
            for doc_id in doc_ids:
                f.write(f"{query_id}\t{doc_id}\t1\n")


def main() -> None:
    """Main entry point for corpus builder CLI."""
    parser = argparse.ArgumentParser(
        description="Build BEIR-compatible corpus from CROP-dataset QA pairs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing JSONL files with QA pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for corpus, queries, and qrels",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs to process (default: process all)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for deduplication (default: 0.95)",
    )

    args = parser.parse_args()

    # Load QA pairs
    print(f"[CROP] Loading QA pairs from {args.input}")
    qa_pairs = load_qa_pairs(args.input, limit=args.limit)
    print(f"[CROP] Loaded {len(qa_pairs)} QA pairs")

    # Extract corpus and mappings
    print(f"[CROP] Extracting corpus and deduplicating (threshold={args.dedup_threshold})")
    corpus, queries, qrels = extract_corpus_and_mappings(
        qa_pairs,
        dedup_threshold=args.dedup_threshold,
    )

    print(f"[CROP] Unique documents: {len(corpus)}")
    print(f"[CROP] Queries: {len(queries)}")
    print(f"[CROP] Query-document pairs: {sum(len(docs) for docs in qrels.values())}")

    # Write outputs
    corpus_path = args.output / "corpus.jsonl"
    queries_path = args.output / "queries.jsonl"
    qrels_path = args.output / "qrels" / "test.tsv"

    print(f"[CROP] Writing corpus to {corpus_path}")
    write_corpus(corpus, corpus_path)

    print(f"[CROP] Writing queries to {queries_path}")
    write_queries(queries, queries_path)

    print(f"[CROP] Writing qrels to {qrels_path}")
    write_qrels(qrels, qrels_path)

    print("[CROP] Corpus building complete!")


if __name__ == "__main__":
    main()
