"""RRF hybrid retriever baseline.

Dense + Sparse를 Reciprocal Rank Fusion(RRF)으로 결합하는 리트리버.
정규화 대신 랭크 기반 점수를 사용해 스코어 스케일 문제를 완화한다.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Base import BaseRetriever


@runtime_checkable
class DenseRetrieverProtocol(Protocol):
    """Dense retriever 프로토콜"""

    docs: List[SourceDoc]

    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


@runtime_checkable
class SparseRetrieverProtocol(Protocol):
    """Sparse retriever 프로토콜 (BM25Store/MiniStore 호환)."""

    ids: List[str]

    def scores(self, q: str): ...


class RRFHybridRetriever(BaseRetriever):
    """RRF 기반 hybrid baseline retriever.

    특징:
    - Dense + Sparse 랭크 기반 결합
    - 고정 가중치: α_dense=0.5, α_sparse=0.5
    - RRF: score = alpha / (k + rank)
    """

    ALPHA_DENSE = 0.5
    ALPHA_SPARSE = 0.5
    RRF_K = 60

    def __init__(self, dense: DenseRetrieverProtocol, sparse: SparseRetrieverProtocol):
        self.dense = dense
        self.sparse = sparse

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        if not getattr(self.dense, "docs", None):
            return []

        expand = max(int(k) * 4, int(k))

        dense_docs = self.dense.search(q, k=expand)
        dense_ranks = {d.id: r for r, d in enumerate(dense_docs)}

        _sims, order = self.sparse.scores(q)
        sparse_ranks: Dict[str, int] = {}
        for r, idx in enumerate(order[: min(expand, len(order))]):
            doc_id = self.sparse.ids[int(idx)]
            sparse_ranks[doc_id] = r

        scores: Dict[str, float] = {}
        for doc_id, rank in dense_ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + float(self.ALPHA_DENSE) / float(self.RRF_K + rank + 1)
        for doc_id, rank in sparse_ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + float(self.ALPHA_SPARSE) / float(self.RRF_K + rank + 1)

        if not scores:
            return []

        doc_by_id = {d.id: d for d in (self.dense.docs or [])}
        out: List[SourceDoc] = []
        for doc_id, _score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_by_id.get(doc_id)
            if doc is None:
                continue
            out.append(doc)
            if len(out) >= k:
                break
        return out

    def __repr__(self) -> str:
        n_docs = len(self.dense.docs) if self.dense.docs else 0
        return f"RRFHybridRetriever(docs={n_docs}, k={self.RRF_K})"
