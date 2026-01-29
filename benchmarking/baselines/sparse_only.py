"""Sparse-only retriever baseline.

TF-IDF 기반 스파스 검색만 사용하는 리트리버.
임베딩 없이 키워드 매칭만 수행.
"""

from __future__ import annotations

from typing import List

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Base import BaseRetriever
from core.Services.Retrieval.Hybrid import SparseRetriever
from core.Models.Schemas import SourceDoc


class SparseOnlyRetriever(BaseRetriever):
    """Sparse-only baseline retriever.

    특징:
    - BM25/TF-IDF 기반 키워드 검색만 사용
    - Dense 임베딩 검색 미사용
    - PathRAG 그래프 검색 미사용
    - 온톨로지 매칭 미사용

    논문 베이스라인:
    - "Sparse만 사용하는 경우" 실험에 활용
    - 전통적인 키워드 기반 검색의 성능 기준선
    """

    def __init__(self, sparse: SparseRetriever):
        """
        Args:
            sparse: Sparse store (BM25Store or MiniStore)
        """
        self.sparse = sparse

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        """쿼리와 유사한 문서를 sparse 점수로 검색.

        Args:
            q: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            상위 k개 유사 문서
        """
        return self.sparse.search(q, k)

    def __repr__(self) -> str:
        return f"SparseOnlyRetriever(docs={len(self.sparse.texts) if self.sparse.texts else 0})"
