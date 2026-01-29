"""Naive hybrid retriever baseline.

Dense + Sparse를 단순 고정 가중치로 결합하는 리트리버.
동적 가중치 조정(DAT) 없이 α=0.5로 고정.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

import numpy as np
import faiss

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Base import BaseRetriever
from core.Services.Retrieval.Hybrid import SparseRetriever
from core.Models.Schemas import SourceDoc


@runtime_checkable
class DenseRetrieverProtocol(Protocol):
    """Dense retriever 프로토콜"""
    docs: List[SourceDoc]
    faiss_index: object

    def encode(self, texts: List[str]) -> np.ndarray: ...
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


class NaiveHybridRetriever(BaseRetriever):
    """Naive hybrid baseline retriever.

    특징:
    - Dense + Sparse 단순 결합
    - 고정 가중치: α_dense=0.5, α_sparse=0.5
    - 동적 가중치 조정(DAT) 미사용
    - PathRAG 그래프 검색 미사용
    - 온톨로지 매칭 미사용
    - 작물 필터링 미사용
    - 중복 제거 미사용

    논문 베이스라인:
    - "단순 하이브리드 검색" 실험에 활용
    - DAT(Dynamic Alpha Tuning) 효과 측정을 위한 대조군
    """

    # 고정 가중치
    ALPHA_DENSE = 0.5
    ALPHA_SPARSE = 0.5

    def __init__(self, dense: DenseRetrieverProtocol, sparse: SparseRetriever):
        """
        Args:
            dense: EmbeddingRetriever 또는 호환되는 dense retriever
            sparse: Sparse store (BM25Store or MiniStore)
        """
        self.dense = dense
        self.sparse = sparse

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        """Dense + Sparse 점수를 고정 가중치로 결합하여 검색.

        Args:
            q: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            상위 k개 문서 (점수 기반 정렬)
        """
        # Dense 검색
        dense_pairs = self._get_dense_scores(q, k * 4)

        # Sparse 검색
        sparse_pairs = self._get_sparse_scores(q, k * 4)

        # 정규화
        d_norm = self._normalize(dense_pairs)
        s_norm = self._normalize(sparse_pairs)

        # 점수 결합
        scores: Dict[int, float] = {}
        id_to_dense = {doc.id: i for i, doc in enumerate(self.dense.docs)}

        for idx, sc in d_norm:
            scores[idx] = scores.get(idx, 0.0) + self.ALPHA_DENSE * sc

        for s_idx, sc in s_norm:
            sid = self.sparse.ids[s_idx]
            didx = id_to_dense.get(sid)
            if didx is not None:
                scores[didx] = scores.get(didx, 0.0) + self.ALPHA_SPARSE * sc

        if not scores:
            # fallback to sparse
            return self.sparse.search(q, k)

        # 점수 기준 정렬
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.dense.docs[i] for i, _ in ranked]

    def _get_dense_scores(self, q: str, n: int) -> List[tuple]:
        """Dense 검색 점수 반환"""
        if not self.dense.docs or self.dense.faiss_index is None:
            return []

        qv = self.dense.encode([q])
        faiss.normalize_L2(qv)
        D, I = self.dense.faiss_index.search(qv, min(n, len(self.dense.docs)))
        return [(int(i), float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]

    def _get_sparse_scores(self, q: str, n: int) -> List[tuple]:
        """Sparse 검색 점수 반환"""
        sims, order = self.sparse.scores(q)
        if order.size == 0:
            return []
        # sims는 이미 order 순서로 정렬된 점수 배열이므로, 위치 기반으로 접근해야 함
        n_results = min(n, len(order))
        return [(int(order[i]), float(sims[i])) for i in range(n_results)]

    def _normalize(self, pairs: List[tuple]) -> List[tuple]:
        """점수 정규화 (min-max)"""
        if not pairs:
            return []
        vals = np.array([s for _, s in pairs], dtype=float)
        if np.allclose(vals.max(), vals.min()):
            vals = np.ones_like(vals)
        else:
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        return [(idx, float(v)) for (idx, _), v in zip(pairs, vals)]

    def __repr__(self) -> str:
        n_docs = len(self.dense.docs) if self.dense.docs else 0
        return f"NaiveHybridRetriever(docs={n_docs}, α_dense={self.ALPHA_DENSE}, α_sparse={self.ALPHA_SPARSE})"
