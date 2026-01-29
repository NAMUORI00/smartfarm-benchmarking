"""LightRAG baseline retriever.

lightrag-hku 라이브러리를 사용한 Dual-Level Graph 기반 검색.

설치: pip install lightrag-hku

참고: LightRAG (EMNLP 2025)
https://github.com/HKUDS/LightRAG
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Services.Retrieval.Base import BaseRetriever
from core.Services.Retrieval.LightRAG import (
    LightRAGRetriever,
    LightRAGRetrieverWithDense,
    create_lightrag_retriever,
)
from core.Models.Schemas import SourceDoc


class LightRAGBaseline(BaseRetriever):
    """LightRAG 기반 baseline retriever.

    lightrag-hku 라이브러리를 사용하여:
    - Dual-Level Graph 자동 구축 (엔티티 + 커뮤니티)
    - 다양한 검색 모드 지원 (naive, local, global, hybrid)
    - 커스텀 LLM/임베딩 지원 (llama.cpp, MiniLM 등)

    HybridDATRetriever 대비:
    - 휴리스틱 파라미터 제거
    - 학술적으로 검증된 구현
    - 커뮤니티 기반 High-Level 검색

    논문 비교:
    - HybridDAT vs LightRAG 직접 비교 실험에 활용
    """

    def __init__(
        self,
        retriever: LightRAGRetriever,
    ):
        """
        Args:
            retriever: LightRAGRetriever 인스턴스
        """
        self.retriever = retriever

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        """쿼리 검색.

        Args:
            q: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            상위 k개 관련 문서
        """
        return self.retriever.search(q, k)

    def query(self, q: str, mode: Optional[str] = None) -> str:
        """LightRAG 쿼리 (LLM 응답 생성).

        Args:
            q: 검색 쿼리
            mode: 검색 모드 (naive, local, global, hybrid)

        Returns:
            LLM 생성 응답
        """
        return self.retriever.query(q, mode)

    def __repr__(self) -> str:
        return f"LightRAGBaseline({self.retriever})"

    @classmethod
    def build_from_docs(
        cls,
        docs: List[SourceDoc],
        working_dir: str | Path = "./lightrag_workdir",
        dense_retriever: Optional[BaseRetriever] = None,
        query_mode: str = "hybrid",
    ) -> "LightRAGBaseline":
        """문서로부터 LightRAG 리트리버 구축.

        Args:
            docs: 원본 문서 리스트
            working_dir: LightRAG 작업 디렉토리
            dense_retriever: Dense retriever (선택, 검색 보완용)
            query_mode: 검색 모드 (naive, local, global, hybrid)

        Returns:
            LightRAGBaseline 인스턴스
        """
        retriever = create_lightrag_retriever(
            working_dir=working_dir,
            docs=docs,
            dense_retriever=dense_retriever,
            query_mode=query_mode,
        )
        return cls(retriever=retriever)

    @classmethod
    def from_working_dir(
        cls,
        working_dir: str | Path,
        docs: Optional[List[SourceDoc]] = None,
        dense_retriever: Optional[BaseRetriever] = None,
        query_mode: str = "hybrid",
    ) -> "LightRAGBaseline":
        """기존 작업 디렉토리에서 로드.

        이미 구축된 그래프가 있는 경우 사용.

        Args:
            working_dir: LightRAG 작업 디렉토리
            docs: 문서 리스트 (검색용, 선택)
            dense_retriever: Dense retriever (선택)
            query_mode: 검색 모드

        Returns:
            LightRAGBaseline 인스턴스
        """
        if dense_retriever is not None:
            retriever = LightRAGRetrieverWithDense(
                working_dir=working_dir,
                dense_retriever=dense_retriever,
                query_mode=query_mode,
            )
        else:
            retriever = LightRAGRetriever(
                working_dir=working_dir,
                query_mode=query_mode,
            )

        retriever.initialize()

        if docs:
            retriever._docs = docs

        return cls(retriever=retriever)

    @property
    def working_dir(self) -> Path:
        """작업 디렉토리 반환."""
        return self.retriever.working_dir

    @property
    def docs(self) -> List[SourceDoc]:
        """문서 리스트 반환."""
        return self.retriever._docs
