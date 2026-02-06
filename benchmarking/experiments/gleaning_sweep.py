from __future__ import annotations

"""Gleaning sweep runner for on-prem KB update extraction.

This experiment measures the effect of `max_gleanings` (0/1/2) on:
  - extraction coverage (entities/relations per chunk)
  - parse drop rate (invalid/mismatched items dropped by validators)
  - downstream tag/graph retrieval hit-rate@K (optional)

The extraction itself uses the on-prem local llama.cpp endpoint via
`core.Services.Ingest.KBUpdateExtractor.extract_kb_update`.

Research references (decision trace):
  - Ref[4]: LightRAG (https://arxiv.org/abs/2410.05779)
  - Ref[17]: Causal extraction survey (https://arxiv.org/abs/2101.06426)
  - Ref[22]: RAGAS (https://arxiv.org/abs/2309.15217)
The sweep intentionally limits gleaning to {0,1,2} for edge cost control.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarking.bootstrap import ensure_search_on_path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _pick_text(row: Dict[str, Any]) -> str:
    for k in ("text", "text_ko", "text_en", "content", "document", "body"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


@dataclass(frozen=True)
class ChunkingConfig:
    method: str
    window_size: int
    window_stride: int
    token_size: int
    token_overlap: int
    tokenizer_model: str


def _chunk_corpus(rows: Sequence[Dict[str, Any]], *, cfg: ChunkingConfig) -> List[Tuple[str, str]]:
    ensure_search_on_path()
    from core.Services.Ingest.Chunking import sentence_split, token_chunks, window_chunks

    out: List[Tuple[str, str]] = []
    method = str(cfg.method or "sentence_window").strip().lower()
    for i, r in enumerate(rows):
        doc_id = str(r.get("id") or r.get("_id") or r.get("doc_id") or f"doc_{i}").strip() or f"doc_{i}"
        text = _pick_text(r)
        if not text.strip():
            continue
        if method == "token":
            chunks = token_chunks(
                text,
                chunk_token_size=int(cfg.token_size),
                chunk_token_overlap=int(cfg.token_overlap),
                tokenizer_model=str(cfg.tokenizer_model or "gpt-4o-mini"),
            )
        else:
            sents = sentence_split(text)
            chunks = window_chunks(sents, window=int(cfg.window_size), stride=int(cfg.window_stride))
        for j, ch in enumerate(chunks):
            chs = (ch or "").strip()
            if not chs:
                continue
            out.append((f"{doc_id}#c{j}", chs))
    return out


def _build_reference_qrels(entities: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Build canonical_id -> [chunk_id] mapping."""
    out: Dict[str, List[str]] = {}
    for chunk_id, canonical_id in entities:
        cid = str(canonical_id or "").strip()
        if not cid:
            continue
        out.setdefault(cid, []).append(str(chunk_id))
    for cid in list(out.keys()):
        out[cid] = sorted(set(out[cid]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gleaning sweep for on-prem KB update extraction")
    ap.add_argument("--input-jsonl", required=True, help="Public corpus JSONL (id,text,metadata...)")
    ap.add_argument("--base-index-dir", default="smartfarm-search/data/index", help="Base bundle dir for alias candidates (tags/graph)")
    ap.add_argument("--out", default="output/gleaning_sweep.json")
    ap.add_argument("--gleanings", default="0,1,2", help="Comma list of max_gleanings values")
    ap.add_argument("--max-chunks", type=int, default=200, help="Cap number of chunks to process")
    ap.add_argument("--timeout-s", type=float, default=60.0)

    ap.add_argument("--chunk-method", default="token", choices=["sentence_window", "token"])
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--chunk-stride", type=int, default=2)
    ap.add_argument("--chunk-token-size", type=int, default=1200)
    ap.add_argument("--chunk-token-overlap", type=int, default=100)
    ap.add_argument("--chunk-tokenizer-model", default="gpt-4o-mini")

    ap.add_argument("--with-downstream", action="store_true", help="Build tags/graph artifacts and evaluate hit@K (K=4)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-queries", type=int, default=200, help="Cap number of canonical-id queries for downstream eval")

    args = ap.parse_args()

    ensure_search_on_path()
    from core.Models.Schemas import SourceDoc
    from core.Services.Ingest.KBUpdateExtractor import BaseKBRegistry, extract_kb_update

    from core.Services.Retrieval.CausalGraph import CausalGraphRetriever
    from core.Services.Retrieval.TagHash import TagHashRetriever
    from core.Services.Retrieval.OverlayIndex import _build_overlay_graph, _build_overlay_tags  # type: ignore

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.input_jsonl))
    cfg = ChunkingConfig(
        method=str(args.chunk_method),
        window_size=int(args.chunk_size),
        window_stride=int(args.chunk_stride),
        token_size=int(args.chunk_token_size),
        token_overlap=int(args.chunk_token_overlap),
        tokenizer_model=str(args.chunk_tokenizer_model),
    )
    chunks = _chunk_corpus(rows, cfg=cfg)
    if int(args.max_chunks) > 0:
        chunks = chunks[: int(args.max_chunks)]

    registry = BaseKBRegistry.load(Path(args.base_index_dir))

    glean_values: List[int] = []
    for p in str(args.gleanings).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            glean_values.append(int(p))
        except Exception:
            continue
    glean_values = sorted(set(max(0, min(2, g)) for g in glean_values)) or [0, 1, 2]

    runs: Dict[str, Any] = {}

    # Research Intent (Ref[4], Ref[17]):
    # compare marginal coverage gains against extraction cost and drop-rate risk.
    # Run extraction for each gleaning setting.
    per_run_entities: Dict[int, List[Tuple[str, str]]] = {}
    per_run_relations: Dict[int, List[Tuple[str, str, str, str]]] = {}
    per_run_docs: Dict[int, List[SourceDoc]] = {}

    for g in glean_values:
        t0 = time.perf_counter()
        entities_rows: List[Tuple[str, str]] = []  # (chunk_id, canonical_id)
        entities_full: List[Tuple[str, str, str, str, float]] = []  # for tag/graph builders
        relations_full: List[Tuple[str, str, str, str, float, str]] = []
        docs: List[SourceDoc] = []

        tot_entities = 0
        tot_relations = 0
        dropped_entities = 0
        dropped_relations = 0
        glean_used: List[int] = []

        for chunk_id, text in chunks:
            docs.append(SourceDoc(id=str(chunk_id), text=str(text), metadata={"source": "gleaning_sweep"}))
            res = extract_kb_update(
                text,
                registry=registry,
                timeout_s=float(args.timeout_s),
                max_gleanings=int(g),
            )
            glean_used.append(int(res.gleaning_passes_used))
            dropped_entities += int(getattr(res, "dropped_entities", 0) or 0)
            dropped_relations += int(getattr(res, "dropped_relations", 0) or 0)

            for e in res.entities:
                tot_entities += 1
                entities_rows.append((str(chunk_id), str(e.canonical_id)))
                entities_full.append((str(chunk_id), str(e.canonical_id), str(e.entity_type), str(e.text), float(e.confidence)))

            for r in res.relations:
                tot_relations += 1
                relations_full.append(
                    (
                        str(chunk_id),
                        str(r.src_canonical_id),
                        str(r.tgt_canonical_id),
                        str(r.relation_type),
                        float(r.confidence),
                        str(r.evidence_text or ""),
                    )
                )

        dt = float(time.perf_counter() - t0)
        per_run_entities[g] = list(entities_rows)
        per_run_relations[g] = [(c, s, t, rt) for (c, s, t, rt, _conf, _ev) in relations_full]
        per_run_docs[g] = list(docs)

        kept_ent = int(tot_entities)
        kept_rel = int(tot_relations)
        drop_ent = int(dropped_entities)
        drop_rel = int(dropped_relations)
        ent_drop_rate = float(drop_ent) / float(max(1, drop_ent + kept_ent))
        rel_drop_rate = float(drop_rel) / float(max(1, drop_rel + kept_rel))

        runs[str(g)] = {
            "max_gleanings": int(g),
            "n_chunks": int(len(chunks)),
            "time_s": float(dt),
            "entities_total": kept_ent,
            "relations_total": kept_rel,
            "entities_per_chunk": float(kept_ent) / float(max(1, len(chunks))),
            "relations_per_chunk": float(kept_rel) / float(max(1, len(chunks))),
            "dropped_entities": drop_ent,
            "dropped_relations": drop_rel,
            "entity_drop_rate": float(ent_drop_rate),
            "relation_drop_rate": float(rel_drop_rate),
            "gleaning_passes_used_hist": {str(x): glean_used.count(x) for x in sorted(set(glean_used))},
        }

    out: Dict[str, Any] = {
        "meta": {
            "input_jsonl": str(args.input_jsonl),
            "base_index_dir": str(args.base_index_dir),
            "chunking": cfg.__dict__,
            "gleanings": glean_values,
            "n_chunks": int(len(chunks)),
        },
        "runs": runs,
    }

    # Optional downstream: build tags/graph and evaluate hit@K against the max-gleaning run as reference.
    if bool(args.with_downstream) and glean_values:
        ref_g = max(glean_values)
        ref_qrels = _build_reference_qrels(per_run_entities.get(ref_g, []))
        qids = sorted(ref_qrels.keys())[: max(0, int(args.max_queries))]

        downstream: Dict[str, Any] = {"reference_gleanings": int(ref_g), "k": int(args.k), "n_queries": int(len(qids)), "by_g": {}}
        for g in glean_values:
            tmp_root = out_path.parent / "gleaning_sweep_artifacts" / f"g{g}"
            if tmp_root.exists():
                # keep old files if present; overwrite for determinism
                pass
            tmp_root.mkdir(parents=True, exist_ok=True)

            docs = per_run_docs.get(g, [])
            docs_by_id = {d.id: d for d in docs}

            # Build artifacts
            _build_overlay_tags(out_dir=tmp_root, docs=docs, entities=[(c, cid, "", cid, 1.0) for c, cid in per_run_entities.get(g, [])])
            _build_overlay_graph(
                out_dir=tmp_root,
                docs=docs,
                entities=[(c, cid, "", cid, 1.0) for c, cid in per_run_entities.get(g, [])],
                relations=[(c, s, t, rt, 1.0, "") for (c, s, t, rt) in per_run_relations.get(g, [])],
            )

            tag = TagHashRetriever(tmp_root / "tags", docs_by_id=docs_by_id)
            graph = CausalGraphRetriever(tmp_root / "graph", docs_by_id=docs_by_id)

            tag_hits = 0
            graph_hits = 0
            for cid in qids:
                relevant = set(ref_qrels.get(cid) or [])
                if not relevant:
                    continue
                q = str(cid)
                tag_res = tag.search(q, k=int(args.k))
                if any(d.id in relevant for d in tag_res):
                    tag_hits += 1
                graph_res = graph.search(q, k=int(args.k))
                if any(d.id in relevant for d in graph_res):
                    graph_hits += 1

            denom = max(1, len(qids))
            downstream["by_g"][str(g)] = {
                "tag_hash_hit_rate": float(tag_hits) / float(denom),
                "causal_graph_hit_rate": float(graph_hits) / float(denom),
            }

        out["downstream"] = downstream

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
