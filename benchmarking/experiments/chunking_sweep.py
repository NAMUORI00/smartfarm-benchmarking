from __future__ import annotations

"""Chunking sweep runner (token chunking) for EdgeKG tuning.

This script runs retrieval-only evaluations for:
  - agxqa (agriculture QA; retrieval track)
  - 2wiki (multi-hop; retrieval track)

and selects a "best" (chunk_token_size, overlap_ratio) config by:
  1) maximizing mean_recall@K(agxqa) + mean_recall@K(2wiki) where K is --k
  2) tie-break: minimizing p95 latency (sum across datasets)
  3) tie-break: minimizing index size (sum across datasets)

The output JSON is designed to be referenced from the paper and validation docs.

Research references (decision trace):
  - Ref[4]: LightRAG (https://arxiv.org/abs/2410.05779)
  - Ref[3]: GraphRAG (https://arxiv.org/abs/2404.16130)
  - Ref[25]: CRAG benchmark framing (https://arxiv.org/abs/2406.04744)
  - Ref[29]: BEIR benchmark framing (https://arxiv.org/abs/2104.08663)
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmarking.experiments.paper_eval import run_paper_eval


@dataclass(frozen=True)
class SweepConfig:
    size: int
    overlap_ratio: float
    overlap: int

    @property
    def key(self) -> str:
        r = int(round(self.overlap_ratio * 100.0))
        return f"tok{self.size}_ov{self.overlap}_r{r:02d}"


def _parse_int_list(raw: str, *, default: Sequence[int]) -> List[int]:
    if not raw:
        return list(default)
    out: List[int] = []
    for p in str(raw).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out or list(default)


def _parse_float_list(raw: str, *, default: Sequence[float]) -> List[float]:
    if not raw:
        return list(default)
    out: List[float] = []
    for p in str(raw).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out or list(default)


def _bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], *, n: int, seed: int) -> Dict[str, float]:
    """Bootstrap CI for mean(a) - mean(b)."""
    xs = [float(x) for x in a]
    ys = [float(y) for y in b]
    if not xs or not ys or len(xs) != len(ys):
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = random.Random(int(seed))
    diffs: List[float] = []
    m = len(xs)
    for _ in range(max(1, int(n))):
        idxs = [rng.randrange(m) for _ in range(m)]
        ma = sum(xs[i] for i in idxs) / float(m)
        mb = sum(ys[i] for i in idxs) / float(m)
        diffs.append(float(ma - mb))
    diffs.sort()
    mean_diff = sum(diffs) / float(len(diffs))
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return {"mean_diff": float(mean_diff), "ci_low": float(lo), "ci_high": float(hi)}


def _extract_recall_per_query(results: Dict[str, Any], *, dataset: str, method: str, k: int) -> List[float]:
    d = (results.get("datasets") or {}).get(dataset) or {}
    m = (d.get("results") or {}).get(method) or {}
    per = m.get("per_query") or []
    key = f"recall@{int(k)}"
    out: List[float] = []
    for r in per:
        mm = (r.get("metrics") or {}) if isinstance(r, dict) else {}
        try:
            out.append(float(mm.get(key, 0.0)))
        except Exception:
            out.append(0.0)
    return out


def _summarize_run(results: Dict[str, Any], *, datasets: Sequence[str], method: str, k: int) -> Dict[str, Any]:
    metric_recall = f"mean_recall@{int(k)}"
    metric_ndcg = f"mean_ndcg@{int(k)}"
    out: Dict[str, Any] = {"datasets": {}}
    for ds in datasets:
        d = (results.get("datasets") or {}).get(ds) or {}
        m = (d.get("results") or {}).get(method) or {}
        retrieval = m.get("retrieval") or {}
        latency = m.get("latency") or {}
        idx_build = d.get("index_build") or {}

        out["datasets"][ds] = {
            metric_recall: float(retrieval.get(metric_recall, 0.0) or 0.0),
            "mean_mrr": float(retrieval.get("mean_mrr", 0.0) or 0.0),
            metric_ndcg: float(retrieval.get(metric_ndcg, 0.0) or 0.0),
            "lat_p95_ms": float(latency.get("p95_ms", 0.0) or 0.0),
            "n_chunks": int(d.get("n_chunks", 0) or 0),
            "index_size_bytes": int(
                (idx_build.get("dense_size_bytes", 0) or 0)
                + (idx_build.get("sparse_size_bytes", 0) or 0)
                + (idx_build.get("trigraph_size_bytes", 0) or 0)
            ),
        }
    return out


def _score_summary(summary: Dict[str, Any], *, datasets: Sequence[str], k: int) -> Tuple[float, float, int]:
    # Research Intent:
    # Ref[29]: BEIR (https://arxiv.org/abs/2104.08663)
    # Ref[25]: CRAG (https://arxiv.org/abs/2406.04744)
    # Use retrieval quality first, then efficiency tie-breakers.
    metric_recall = f"mean_recall@{int(k)}"
    score = 0.0
    p95_sum = 0.0
    size_sum = 0
    for ds in datasets:
        block = (summary.get("datasets") or {}).get(ds) or {}
        score += float(block.get(metric_recall, 0.0) or 0.0)
        p95_sum += float(block.get("lat_p95_ms", 0.0) or 0.0)
        size_sum += int(block.get("index_size_bytes", 0) or 0)
    return float(score), float(p95_sum), int(size_sum)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chunking sweep (token chunking) for retrieval-only tuning")
    ap.add_argument("--out", default="output/chunking_sweep.json")
    ap.add_argument("--datasets", default="agxqa,2wiki", help="Comma list from {agxqa,2wiki}")
    ap.add_argument("--method", default="ours_full", help="Retriever method name (default: ours_full)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-model-id", default="", help="Optional embedding model override (e.g., minilm)")
    ap.add_argument("--sizes", default="256,512,768,1024,1200,1536")
    ap.add_argument("--overlap-ratios", default="0,0.08,0.12")
    ap.add_argument("--tokenizer-model", default="gpt-4o-mini")
    ap.add_argument("--bootstraps", type=int, default=500)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip().lower() for d in str(args.datasets).split(",") if d.strip()]
    datasets = [d for d in datasets if d in {"agxqa", "2wiki"}]
    if not datasets:
        datasets = ["agxqa", "2wiki"]

    sizes = _parse_int_list(str(args.sizes), default=[256, 512, 768, 1024, 1200, 1536])
    ratios = _parse_float_list(str(args.overlap_ratios), default=[0.0, 0.08, 0.12])
    tokenizer_model = str(args.tokenizer_model or "gpt-4o-mini")
    primary_metric = f"mean_recall@{int(args.k)}"

    grid: List[SweepConfig] = []
    for s in sizes:
        s = max(1, int(s))
        for r in ratios:
            r = max(0.0, min(0.95, float(r)))
            ov = int(round(float(s) * float(r)))
            ov = max(0, min(int(s) - 1, ov))
            grid.append(SweepConfig(size=int(s), overlap_ratio=float(r), overlap=int(ov)))
    # Always include paper baselines (exact overlaps) even if they don't align with the ratio grid.
    # Baseline A (LightRAG-compatible): 1200/100, Baseline B (small-chunk sanity): 512/64
    for s, ov in [(1200, 100), (512, 64)]:
        s = max(1, int(s))
        ov = max(0, min(s - 1, int(ov)))
        r = float(ov) / float(s) if s else 0.0
        grid.append(SweepConfig(size=int(s), overlap_ratio=float(r), overlap=int(ov)))

    # Deduplicate configs by (size, overlap) while keeping stable order.
    dedup: List[SweepConfig] = []
    seen_pairs: set[tuple[int, int]] = set()
    for cfg in grid:
        key = (int(cfg.size), int(cfg.overlap))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        dedup.append(cfg)
    grid = dedup

    selection_rule = {
        "priority_1": f"maximize sum({primary_metric}) over datasets",
        "priority_2": "minimize tie_p95_ms_sum",
        "priority_3": "minimize tie_index_size_sum",
    }
    meta = {
        "datasets": list(datasets),
        "method": str(args.method),
        "k": int(args.k),
        "primary_metric": primary_metric,
        "selection_rule": selection_rule,
        "research_basis_refs": [
            "Ref[4] https://arxiv.org/abs/2410.05779",
            "Ref[3] https://arxiv.org/abs/2404.16130",
            "Ref[29] https://arxiv.org/abs/2104.08663",
            "Ref[25] https://arxiv.org/abs/2406.04744",
        ],
        "max_queries": int(args.max_queries),
        "seed": int(args.seed),
        "embed_model_id": str(args.embed_model_id or ""),
        "grid": [{"size": c.size, "overlap_ratio": c.overlap_ratio, "overlap": c.overlap, "key": c.key} for c in grid],
    }

    records: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_score: Tuple[float, float, int] | None = None

    for cfg in grid:
        run_dir = out_path.parent / "chunking_sweep_runs" / cfg.key
        run_dir.mkdir(parents=True, exist_ok=True)

        results = run_paper_eval(
            datasets=datasets,
            methods=[str(args.method)],
            k=int(args.k),
            out_path=run_dir / "paper_eval.json",
            max_queries=int(args.max_queries),
            seed=int(args.seed),
            embed_model_id=(str(args.embed_model_id).strip() or None),
            chunk_method="token",
            chunk_token_size=int(cfg.size),
            chunk_token_overlap=int(cfg.overlap),
            chunk_tokenizer_model=tokenizer_model,
            retrieval_only=True,
            with_ragas=False,
            ragas_max_queries=0,
            ragas_model="",
            ragas_base_url="",
            ragas_api_key="",
        )

        summary = _summarize_run(results, datasets=datasets, method=str(args.method), k=int(args.k))
        score, p95_sum, size_sum = _score_summary(summary, datasets=datasets, k=int(args.k))

        rec = {
            "key": cfg.key,
            "chunk_method": "token",
            "chunk_token_size": int(cfg.size),
            "chunk_token_overlap": int(cfg.overlap),
            "overlap_ratio": float(cfg.overlap_ratio),
            "summary": summary,
            "score": float(score),
            "score_metric": f"sum({primary_metric})",
            "tie_p95_ms_sum": float(p95_sum),
            "tie_index_size_sum": int(size_sum),
        }
        records.append(rec)

        cur_score = (float(score), float(p95_sum), int(size_sum))
        if best is None:
            best = rec
            best_score = cur_score
            continue
        assert best_score is not None
        # Higher score better; lower p95 and size better.
        if cur_score[0] > best_score[0]:
            best, best_score = rec, cur_score
        elif cur_score[0] == best_score[0] and cur_score[1] < best_score[1]:
            best, best_score = rec, cur_score
        elif cur_score[0] == best_score[0] and cur_score[1] == best_score[1] and cur_score[2] < best_score[2]:
            best, best_score = rec, cur_score

    out: Dict[str, Any] = {"meta": meta, "records": records, "best": best}

    # Optional bootstrap CI when the best config is very small (e.g., 256 tokens).
    if best and int(best.get("chunk_token_size", 0) or 0) == 256:
        br = float(best.get("overlap_ratio", 0.0) or 0.0)
        boot: Dict[str, Any] = {
            "note": f"bootstrap CI for recall@{int(args.k)} diff when best uses 256 tokens",
            "comparisons": {},
        }
        for target_size in (512, 768):
            cand = next(
                (
                    r
                    for r in records
                    if int(r.get("chunk_token_size", 0) or 0) == int(target_size)
                    and abs(float(r.get("overlap_ratio", 0.0) or 0.0) - br) < 1e-9
                ),
                None,
            )
            if cand is None:
                continue

            comp_key = f"256_vs_{target_size}_r{int(round(br * 100.0)):02d}"
            boot["comparisons"][comp_key] = {}
            # Re-read per-query recall arrays from the per-config run output.
            best_run = out_path.parent / "chunking_sweep_runs" / str(best["key"]) / "paper_eval.json"
            cand_run = out_path.parent / "chunking_sweep_runs" / str(cand["key"]) / "paper_eval.json"
            try:
                best_json = json.loads(best_run.read_text(encoding="utf-8"))
                cand_json = json.loads(cand_run.read_text(encoding="utf-8"))
            except Exception:
                continue

            for ds in datasets:
                a = _extract_recall_per_query(best_json, dataset=ds, method=str(args.method), k=int(args.k))
                b = _extract_recall_per_query(cand_json, dataset=ds, method=str(args.method), k=int(args.k))
                boot["comparisons"][comp_key][ds] = _bootstrap_mean_diff(
                    a,
                    b,
                    n=int(args.bootstraps),
                    seed=int(args.seed),
                )
        out["bootstrap"] = boot

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
