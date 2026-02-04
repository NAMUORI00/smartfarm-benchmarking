from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np


def _fake_encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:  # noqa: ARG001
    dim = 16
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256((t or "").encode("utf-8")).digest()
        v = np.frombuffer(h, dtype=np.uint8)[:dim].astype(np.float32)
        out[i, :] = v
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out / norms


def _fake_generate_json(prompt: str) -> dict:  # noqa: ARG001
    return {"answer": "dummy"}


class TestPaperEvalSmoke(unittest.TestCase):
    def test_multihop_all_methods_runs(self) -> None:
        from benchmarking.experiments import paper_eval

        methods = ["bm25_only", "dense_only", "rrf", "trigraph_only", "ours_full"]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "paper_eval.json"

            with patch.object(paper_eval.EmbeddingRetriever, "encode", _fake_encode), patch.object(
                paper_eval, "generate_json", _fake_generate_json
            ):
                result = paper_eval.run_paper_eval(
                    track="multihop",
                    methods=methods,
                    k=4,
                    out_path=out_path,
                    beir_max_queries=5,
                    beir_doc_limit=100,
                    beir_split="test",
                    embed_model_id="dummy",
                    with_ragas=False,
                    ragas_model="dummy",
                    ragas_base_url="",
                    ragas_api_key="",
                )

        self.assertIn("datasets", result)
        self.assertIn("multihop", result["datasets"])
        block = result["datasets"]["multihop"]
        self.assertIn("results", block)

        for m in methods:
            self.assertIn(m, block["results"])
            mblock = block["results"][m]
            self.assertIn("retrieval", mblock)
            self.assertIn("mean_mrr", mblock["retrieval"])
            self.assertIn("mean_precision@4", mblock["retrieval"])
            self.assertIn("qa", mblock)
            self.assertIn("mean_exact_match", mblock["qa"])
            self.assertIn("mean_f1", mblock["qa"])


if __name__ == "__main__":
    unittest.main()

