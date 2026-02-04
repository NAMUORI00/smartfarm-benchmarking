from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import patch

from benchmarking.datasets.types import EvalSample


class _Doc:
    def __init__(self, doc_id: str, text: str, meta: Dict[str, Any] | None = None):
        self.id = doc_id
        self.text = text
        self.metadata = meta or {}


class _FakeDense:
    def __init__(self, docs: List[_Doc]):
        self.docs = docs
        self.faiss_index = object()

    def search(self, q: str, k: int = 4) -> List[_Doc]:  # noqa: ARG002
        return list(self.docs)[:k]


class _FakeSparse:
    def __init__(self, docs: List[_Doc]):
        self.ids = [d.id for d in docs]
        self._docs = docs

    def scores(self, q: str):  # noqa: ARG002
        # Return a deterministic ranking: 0..n-1
        import numpy as np

        n = len(self._docs)
        scores = np.linspace(1.0, 0.0, num=n, dtype=float)
        order = np.arange(n, dtype=int)
        return scores, order

    def search(self, q: str, k: int = 4) -> List[_Doc]:  # noqa: ARG002
        return list(self._docs)[:k]


class _FakeTriGraph:
    def __init__(self, docs: List[_Doc]):
        self._docs = docs

    def search(self, q: str, k: int = 4) -> List[_Doc]:  # noqa: ARG002
        return list(reversed(self._docs))[:k]


def _fake_agxqa() -> Tuple[List[EvalSample], List[Tuple[str, str]]]:
    corpus = [
        ("d1", "Chemigation valves protect both surface and ground water sources."),
        ("d2", "Maintain greenhouse temperature above 10C at night for tomatoes."),
    ]
    samples = [
        EvalSample(qid="q1", question="What do Chemigation valves protect?", relevant_doc_ids={"d1"}, relevance_scores={"d1": 1.0}, gold_answer="both surface and ground water sources"),
        EvalSample(qid="q2", question="What is the optimal tomato night temperature?", relevant_doc_ids={"d2"}, relevance_scores={"d2": 1.0}, gold_answer="above 10c"),
    ]
    return samples, corpus


def _fake_2wiki() -> Tuple[List[EvalSample], List[Tuple[str, str]]]:
    corpus = [
        ("t1", "Title1\nSentence A."),
        ("t2", "Title2\nSentence B."),
    ]
    samples = [
        EvalSample(qid="m1", question="Multi-hop question?", relevant_doc_ids={"t1", "t2"}, relevance_scores={"t1": 1.0, "t2": 1.0}, gold_answer="answer"),
    ]
    return samples, corpus


def _fake_chunk_docs(docs: Sequence[Tuple[str, str]], *, chunk_size: int, chunk_stride: int):  # noqa: ARG001
    # One chunk per doc for the smoke test.
    out: List[_Doc] = []
    for doc_id, text in docs:
        out.append(_Doc(f"{doc_id}#c0", text, {"doc_id": doc_id, "chunk_index": 0}))
    return out


def _fake_build_dense_sparse_indices(docs, *, out_dir: Path, embed_model_id: str | None, need_dense: bool, need_sparse: bool):  # noqa: ARG001
    dense = _FakeDense(list(docs))
    sparse = _FakeSparse(list(docs))
    stats = {
        "dense_build_s": 0.0,
        "dense_size_bytes": 0,
        "dense_model_id": "dummy",
        "dense_dim": 16,
        "sparse_build_s": 0.0,
        "sparse_size_bytes": 0,
    }
    return dense, sparse, stats


def _fake_build_trigraph_index(docs, *, out_dir: Path, embedder):  # noqa: ARG001
    trigraph = _FakeTriGraph(list(docs))
    stats = {"trigraph_build_s": 0.0, "trigraph_size_bytes": 0}
    return trigraph, stats


def _fake_evaluate_with_generation(*, method_name: str, retriever, samples: Sequence[EvalSample], k: int, k_values: Sequence[int]):  # noqa: ARG001
    from benchmarking.experiments.paper_eval import _evaluate_retrieval_only

    block = _evaluate_retrieval_only(method_name=method_name, retriever=retriever, samples=samples, k=k, k_values=k_values)
    block["qa"] = {"mean_exact_match": 0.0, "mean_f1": 0.0}
    return block


class TestPaperEvalSmoke(unittest.TestCase):
    def test_public_datasets_all_methods_runs(self) -> None:
        from benchmarking.experiments import paper_eval

        methods = ["bm25_only", "dense_only", "rrf", "trigraph_only", "ours_full"]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "paper_eval.json"

            with patch.object(paper_eval, "load_agxqa", side_effect=lambda **_: _fake_agxqa()), patch.object(
                paper_eval, "load_twowiki_multihopqa", side_effect=lambda **_: _fake_2wiki()
            ), patch.object(
                paper_eval,
                "_chunk_docs",
                side_effect=_fake_chunk_docs,
            ), patch.object(
                paper_eval,
                "_build_dense_sparse_indices",
                side_effect=_fake_build_dense_sparse_indices,
            ), patch.object(
                paper_eval,
                "_build_trigraph_index",
                side_effect=_fake_build_trigraph_index,
            ), patch.object(
                paper_eval,
                "_evaluate_with_generation",
                side_effect=_fake_evaluate_with_generation,
            ):
                result = paper_eval.run_paper_eval(
                    datasets=["agxqa", "2wiki"],
                    methods=methods,
                    k=4,
                    out_path=out_path,
                    max_queries=3,
                    seed=42,
                    embed_model_id="dummy",
                    with_ragas=False,
                    ragas_max_queries=2,
                    ragas_model="dummy",
                    ragas_base_url="",
                    ragas_api_key="",
                )

        self.assertIn("datasets", result)
        self.assertIn("agxqa", result["datasets"])
        self.assertIn("2wiki", result["datasets"])

        agxqa = result["datasets"]["agxqa"]
        self.assertEqual(agxqa["mode"], "qa")
        self.assertIn("results", agxqa)
        for m in methods:
            self.assertIn(m, agxqa["results"])
            mblock = agxqa["results"][m]
            self.assertIn("retrieval", mblock)
            self.assertIn("mean_mrr", mblock["retrieval"])
            self.assertIn("mean_precision@4", mblock["retrieval"])
            self.assertIn("qa", mblock)
            self.assertIn("mean_exact_match", mblock["qa"])
            self.assertIn("mean_f1", mblock["qa"])

        twowiki = result["datasets"]["2wiki"]
        self.assertEqual(twowiki["mode"], "retrieval")
        self.assertIn("results", twowiki)
        for m in methods:
            self.assertIn(m, twowiki["results"])
            mblock = twowiki["results"][m]
            self.assertIn("retrieval", mblock)
            self.assertIn("mean_mrr", mblock["retrieval"])
            self.assertIn("mean_precision@4", mblock["retrieval"])


if __name__ == "__main__":
    unittest.main()
