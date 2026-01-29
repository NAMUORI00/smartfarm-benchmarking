"""RRF hybrid retriever baseline.

Dense + Sparse를 Reciprocal Rank Fusion(RRF)으로 결합하는 리트리버.
정규화 대신 랭크 기반 점수를 사용해 스코어 스케일 문제를 완화한다.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

import faiss
import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Base import BaseRetriever
from core.Services.Retrieval.Hybrid import SparseRetriever


@runtime_checkable
class DenseRetrieverProtocol(Protocol):
    """Dense retriever 프로토콜"""

    docs: List[SourceDoc]
    faiss_index: object

    def encode(self, texts: List[str]) -> np.ndarray: ...
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


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

    def __init__(self, dense: DenseRetrieverProtocol, sparse: SparseRetriever):
        self.dense = dense
        self.sparse = sparse

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        dense_pairs = self._get_dense_scores(q, k * 4)
        sparse_pairs = self._get_sparse_scores(q, k * 4)

        scores: Dict[int, float] = {}
        id_to_dense = {doc.id: i for i, doc in enumerate(self.dense.docs)}

        for rank, (idx, _) in enumerate(dense_pairs):
            rrf_score = self.ALPHA_DENSE / (self.RRF_K + rank + 1)
            scores[idx] = scores.get(idx, 0.0) + rrf_score

        for rank, (s_idx, _) in enumerate(sparse_pairs):
            sid = self.sparse.ids[s_idx]
            didx = id_to_dense.get(sid)
            if didx is not None:
                rrf_score = self.ALPHA_SPARSE / (self.RRF_K + rank + 1)
                scores[didx] = scores.get(didx, 0.0) + rrf_score

        if not scores:
            return self.sparse.search(q, k)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.dense.docs[i] for i, _ in ranked]

    def _get_dense_scores(self, q: str, n: int) -> List[tuple]:
        if not self.dense.docs or self.dense.faiss_index is None:
            return []

        qv = self.dense.encode([q])
        faiss.normalize_L2(qv)
        D, I = self.dense.faiss_index.search(qv, min(n, len(self.dense.docs)))
        return [(int(i), float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]

    def _get_sparse_scores(self, q: str, n: int) -> List[tuple]:
        sims, order = self.sparse.scores(q)
        if order.size == 0:
            return []
        # sims는 이미 order 순서로 정렬된 점수 배열이므로, 위치 기반으로 접근해야 함
        n_results = min(n, len(order))
        return [(int(order[i]), float(sims[i])) for i in range(n_results)]

    def __repr__(self) -> str:
        n_docs = len(self.dense.docs) if self.dense.docs else 0
        return f"RRFHybridRetriever(docs={n_docs}, k={self.RRF_K})"
