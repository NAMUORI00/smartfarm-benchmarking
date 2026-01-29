#!/usr/bin/env python3
"""PathRAG-lt RAGAS Evaluation on BEIR Datasets

Compares PathRAG-lt against baselines using RAGAS LLM-as-Judge metrics.
Supports BEIR datasets (corpus.jsonl, queries.jsonl, qrels/*.tsv format).
Uses DeepSeek-V3 for evaluation.

NOTE: BEIR format limitations for RAGAS evaluation
=========================================================================
BEIR datasets lack reference answers, limiting RAGAS evaluation:
- ✅ Context Precision/Recall metrics (using qrels as ground truth)
- ❌ Answer Relevancy/Correctness (no reference answers available)
- ❌ Faithfulness (no generated answers to evaluate)

For full RAGAS evaluation with answer quality metrics, use:
- ragas_eval.py with datasets that include reference answers
- crop_pathrag_eval.py with --with-ragas for context-focused evaluation

This script evaluates retrieval quality (context metrics) only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Set API credentials before importing RAGAS
os.environ.setdefault("OPENAI_BASE_URL", "https://nano-gpt.com/api/v1")

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import context_precision, context_recall, answer_relevancy
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Embeddings import EmbeddingRetriever
from core.Services.Retrieval.Sparse import BM25Store
from benchmarking.baselines import (
    DenseOnlyRetriever,
    RRFHybridRetriever,
    PathRAGHybridRetriever,
)


def load_corpus(corpus_path: Path) -> List[SourceDoc]:
    """Load corpus from BEIR corpus.jsonl format.

    Args:
        corpus_path: Path to BEIR corpus.jsonl file

    Returns:
        List of SourceDoc objects
    """
    docs: List[SourceDoc] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = data.get("_id") or data.get("id")
            title = data.get("title", "")
            text = data.get("text", "")
            merged_text = "\n".join([part for part in [title, text] if part])
            docs.append(SourceDoc(
                id=str(doc_id),
                text=merged_text,
                metadata={"title": title}
            ))
    print(f"[Corpus] Loaded {len(docs)} documents from {corpus_path}")
    return docs


def load_qa_dataset(dataset_dir: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load BEIR queries and qrels as QA items.

    Args:
        dataset_dir: Path to BEIR dataset directory
        limit: Optional limit on number of items

    Returns:
        List of QA items with id, question, answer, source_ids
    """
    import csv

    # Load queries from queries.jsonl
    queries = {}
    queries_path = dataset_dir / "queries.jsonl"
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query_id = data.get("_id") or data.get("id")
            query_text = data.get("text") or data.get("query", "")
            queries[str(query_id)] = query_text

    # Load qrels from qrels/test.tsv (or dev.tsv)
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        qrels_path = dataset_dir / "qrels" / "dev.tsv"

    qrels = {}  # query_id -> list of relevant doc_ids
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query_id = row.get("query-id") or row.get("qid")
            doc_id = row.get("corpus-id") or row.get("doc-id")
            score = float(row.get("score") or row.get("rel", 0))
            if score > 0:
                qrels.setdefault(str(query_id), []).append(str(doc_id))

    # Build QA items
    qa_items = []
    for query_id, query_text in queries.items():
        if query_id not in qrels:
            continue
        qa_items.append({
            "id": query_id,
            "question": query_text,
            "answer": "",  # BEIR doesn't have reference answers
            "source_ids": qrels[query_id],
        })
        if limit and len(qa_items) >= limit:
            break

    print(f"[QA Dataset] Loaded {len(qa_items)} questions from {dataset_dir}")
    return qa_items


def build_retrievers(docs: List[SourceDoc]) -> Dict[str, Any]:
    """Build retriever instances for all modes.

    Args:
        docs: Document corpus

    Returns:
        Dictionary of retriever instances by mode name
    """
    print("[Setup] Building retrievers...")

    # Dense retriever (FAISS + embeddings)
    dense = EmbeddingRetriever()
    dense.build(docs)

    # Sparse retriever (BM25)
    sparse = BM25Store()
    sparse.index(docs)

    # Build baselines
    dense_only = DenseOnlyRetriever(dense)
    rrf_hybrid = RRFHybridRetriever(dense, sparse)
    pathrag_hybrid = PathRAGHybridRetriever(dense, docs)

    retrievers = {
        "dense_only": dense_only,
        "rrf_hybrid": rrf_hybrid,
        "pathrag_hybrid": pathrag_hybrid,
    }

    print(f"[Setup] Built {len(retrievers)} retrievers")
    return retrievers


def evaluate_mode(
    mode_name: str,
    retriever: Any,
    qa_items: List[Dict[str, Any]],
    llm: Any,
    embeddings: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a single retrieval mode using RAGAS.

    Args:
        mode_name: Name of retrieval mode
        retriever: Retriever instance
        qa_items: QA dataset items
        llm: RAGAS LLM instance
        embeddings: RAGAS embeddings instance
        top_k: Number of documents to retrieve

    Returns:
        Dictionary with mode name and RAGAS scores
    """
    print(f"\n[{mode_name}] Evaluating with RAGAS...")

    # Build RAGAS dataset
    ragas_rows = []
    for item in qa_items:
        question = item["question"]
        source_ids = item["source_ids"]

        # Retrieve documents
        retrieved_docs = retriever.search(question, k=top_k)
        contexts = [doc.text for doc in retrieved_docs]

        # Build ground_truth from source_ids (relevant doc IDs)
        # RAGAS context_precision/recall use ground_truth to check if retrieved contexts are relevant
        ground_truth = "\n".join(source_ids)  # Join relevant doc IDs as ground truth

        ragas_rows.append({
            "question": question,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in ragas_rows],
        "contexts": [r["contexts"] for r in ragas_rows],
        "ground_truth": [r["ground_truth"] for r in ragas_rows],
    })

    # Run RAGAS evaluation with only context metrics (no answer metrics for BEIR)
    metrics = [context_precision, context_recall]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )

    # Extract scores
    scores = {}
    if hasattr(result, "scores"):
        # Aggregate scores across all examples
        for metric_name in ["context_precision", "context_recall"]:
            values = [row.get(metric_name) for row in result.scores if row.get(metric_name) is not None]
            if values:
                scores[metric_name] = sum(values) / len(values)
    else:
        scores = dict(result)

    print(f"[{mode_name}] Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")

    return {
        "mode": mode_name,
        "num_queries": len(qa_items),
        "top_k": top_k,
        "scores": scores,
    }


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print comparison table of RAGAS scores across modes.

    Args:
        results: List of result dictionaries from evaluate_mode
    """
    print("\n" + "="*80)
    print("RAGAS EVALUATION COMPARISON (BEIR Dataset)")
    print("="*80)
    print(f"{'Mode':<20} {'Context Precision':<30} {'Context Recall':<30}")
    print("-"*80)

    for result in results:
        mode = result["mode"]
        scores = result["scores"]
        precision = scores.get("context_precision", 0.0)
        recall = scores.get("context_recall", 0.0)
        print(f"{mode:<20} {precision:<30.4f} {recall:<30.4f}")

    print("="*80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PathRAG-lt RAGAS Evaluation - Compare retrievers using LLM-as-Judge metrics on BEIR datasets"
    )
    parser.add_argument(
        "--beir-dir",
        type=Path,
        default=Path("data/beir"),
        help="Path to BEIR datasets directory (default: data/beir)",
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        help="BEIR dataset name (default: scifact)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pathrag_ragas_results.json"),
        help="Output path for results JSON (default: output/pathrag_ragas_results.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of QA items to evaluate (default: 50)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    parser.add_argument(
        "--doc-limit",
        type=int,
        default=5000,
        help="Limit number of documents to index (default: 5000, None for all)",
    )
    args = parser.parse_args()

    # Resolve BEIR dataset paths
    dataset_dir = args.beir_dir / args.dataset
    corpus_path = dataset_dir / "corpus.jsonl"

    if not corpus_path.exists():
        print(f"[Error] Corpus not found at {corpus_path}")
        print(f"[Error] Please ensure BEIR dataset '{args.dataset}' is downloaded to {dataset_dir}")
        return

    # Load data
    docs = load_corpus(corpus_path)
    if args.doc_limit:
        docs = docs[:args.doc_limit]
        print(f"[Corpus] Limited to {len(docs)} documents")

    qa_items = load_qa_dataset(dataset_dir, limit=args.limit)

    # Build retrievers
    retrievers = build_retrievers(docs)

    # Setup RAGAS LLM and embeddings
    print("\n[Setup] Configuring RAGAS evaluation with DeepSeek-V3...")
    client = OpenAI(
        base_url="https://nano-gpt.com/api/v1",
        api_key="sk-nano-8c197b75-1c2f-4da8-87c0-06d917db6a9c",
    )
    llm = llm_factory(model="deepseek-v3-0324", client=client)

    # Use lightweight multilingual embeddings for RAGAS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    # Evaluate all modes
    results = []
    for mode_name, retriever in retrievers.items():
        mode_result = evaluate_mode(
            mode_name=mode_name,
            retriever=retriever,
            qa_items=qa_items,
            llm=llm,
            embeddings=embeddings,
            top_k=args.top_k,
        )
        results.append(mode_result)

    # Print comparison table
    print_comparison_table(results)

    # Save results
    output_payload = {
        "beir_dataset": args.dataset,
        "dataset_dir": str(dataset_dir),
        "corpus_path": str(corpus_path),
        "num_documents": len(docs),
        "num_queries": len(qa_items),
        "top_k": args.top_k,
        "llm_model": "deepseek-v3-0324",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Results saved to {args.output}")


if __name__ == "__main__":
    main()
