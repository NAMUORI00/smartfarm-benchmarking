from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _fake_run_paper_eval(**kwargs):  # type: ignore[no-untyped-def]
    size = int(kwargs.get("chunk_token_size", 0) or 0)
    method = str((kwargs.get("methods") or ["ours_full"])[0])
    datasets = list(kwargs.get("datasets") or ["agxqa", "2wiki"])

    # Make selection outcome depend on recall@10, not recall@4.
    if size == 1200:
        recall4 = 0.2
        recall10 = 0.9
        p95 = 120.0
        n10 = 0.7
    else:
        recall4 = 0.8
        recall10 = 0.1
        p95 = 90.0
        n10 = 0.2

    ds_block = {}
    for ds in datasets:
        ds_block[str(ds)] = {
            "n_chunks": 10,
            "index_build": {
                "dense_size_bytes": 100,
                "sparse_size_bytes": 50,
                "trigraph_size_bytes": 25,
            },
            "results": {
                method: {
                    "retrieval": {
                        "mean_recall@4": recall4,
                        "mean_recall@10": recall10,
                        "mean_mrr": 0.5,
                        "mean_ndcg@4": 0.3,
                        "mean_ndcg@10": n10,
                    },
                    "latency": {"p95_ms": p95},
                    "per_query": [
                        {"metrics": {"recall@4": recall4, "recall@10": recall10}},
                        {"metrics": {"recall@4": recall4, "recall@10": recall10}},
                    ],
                }
            },
        }
    return {"datasets": ds_block}


class TestChunkingSweepKConsistency(unittest.TestCase):
    def test_k10_selection_uses_recall10(self) -> None:
        from benchmarking.experiments import chunking_sweep

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "chunking_sweep.json"
            argv = [
                "chunking_sweep.py",
                "--out",
                str(out_path),
                "--k",
                "10",
                "--sizes",
                "512",
                "--overlap-ratios",
                "0",
                "--max-queries",
                "2",
                "--datasets",
                "agxqa,2wiki",
                "--method",
                "ours_full",
            ]

            with patch.object(chunking_sweep, "run_paper_eval", side_effect=_fake_run_paper_eval), patch(
                "sys.argv", argv
            ):
                rc = chunking_sweep.main()
                self.assertEqual(rc, 0)

            raw = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["meta"]["k"], 10)
            self.assertEqual(raw["meta"]["primary_metric"], "mean_recall@10")
            self.assertIn("selection_rule", raw["meta"])
            self.assertIn("research_basis_refs", raw["meta"])
            self.assertEqual(int(raw["best"]["chunk_token_size"]), 1200)
