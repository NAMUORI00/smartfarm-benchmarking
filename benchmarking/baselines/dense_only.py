"""Dense-only retriever baseline.

임베딩 유사도만 사용하는 리트리버.
BM25/TF-IDF 없이 순수 semantic search만 수행.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Base import BaseRetriever
from core.Models.Schemas import SourceDoc


@runtime_checkable
class DenseRetrieverProtocol(Protocol):
    """Dense retriever 프로토콜"""
    docs: List[SourceDoc]
    faiss_index: object

    def encode(self, texts: List[str]) -> np.ndarray: ...
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


class DenseOnlyRetriever(BaseRetriever):
    """Dense-only baseline retriever.

    특징:
    - FAISS 기반 임베딩 유사도 검색만 사용
    - BM25/TF-IDF 스파스 검색 미사용
    - PathRAG 그래프 검색 미사용
    - 온톨로지 매칭 미사용

    논문 베이스라인:
    - "Dense retrieval만 사용하는 경우" 실험에 활용
    """

    def __init__(self, dense: DenseRetrieverProtocol):
        """
        Args:
            dense: EmbeddingRetriever 또는 호환되는 dense retriever
        """
        self.dense = dense

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        """쿼리와 유사한 문서를 임베딩 유사도로 검색.

        Args:
            q: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            상위 k개 유사 문서
        """
        if not self.dense.docs or self.dense.faiss_index is None:
            return []

        return self.dense.search(q, k)

    def __repr__(self) -> str:
        return f"DenseOnlyRetriever(docs={len(self.dense.docs) if self.dense.docs else 0})"
