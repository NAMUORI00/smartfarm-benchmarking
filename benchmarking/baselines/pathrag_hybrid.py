"""PathRAG+Dense hybrid retriever baseline.

Dense retrieval로 초기 후보를 찾고, PathRAG 그래프 탐색으로 재랭킹.
온톨로지 기반 룰베이스 그래프 빌더와 BFS 경로 탐색 사용.
"""
from __future__ import annotations

from typing import List, Protocol, Set, runtime_checkable

import numpy as np

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Graph import SmartFarmGraph
from core.Models.Schemas import SourceDoc
from core.Services.Ingest.GraphBuilder import build_graph_from_docs
from core.Services.Retrieval.Base import BaseRetriever
from core.Services.Retrieval.PathScoring import bfs_weighted_paths, balanced_path_selection


@runtime_checkable
class DenseRetrieverProtocol(Protocol):
    """Dense retriever 프로토콜"""
    docs: List[SourceDoc]
    faiss_index: object

    def encode(self, texts: List[str]) -> np.ndarray: ...
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


class PathRAGHybridRetriever(BaseRetriever):
    """PathRAG+Dense hybrid baseline retriever.

    특징:
    - Stage 1: Dense retrieval로 초기 후보 검색 (k * 2)
    - Stage 2: 온톨로지 기반 룰베이스 그래프 생성
    - Stage 3: PathRAG BFS 경로 탐색으로 관련 노드 발견
    - Stage 4: RRF dense score + path score 결합으로 최종 랭킹

    논문 베이스라인:
    - "PathRAG 그래프 탐색 + Dense retrieval 결합" 실험에 활용
    - 온톨로지 매칭 + 인과관계 엣지로 경로 기반 재랭킹
    """

    def __init__(
        self,
        dense: DenseRetrieverProtocol,
        docs: List[SourceDoc],
        use_llm_graph: bool = False
    ):
        """
        Args:
            dense: EmbeddingRetriever 또는 호환되는 dense retriever
            docs: 전체 문서 리스트 (그래프 빌드용)
            use_llm_graph: LLM 기반 그래프 빌더 사용 여부 (현재 미지원, 룰베이스만)
        """
        self.dense = dense
        self.docs = docs
        self.use_llm_graph = use_llm_graph

        # 룰베이스 그래프 빌더로 그래프 생성
        self.graph = build_graph_from_docs(docs)

        # Practice node IDs 추출 (문서 ID와 매칭)
        self.practice_node_ids: Set[str] = {
            node_id for node_id, node in self.graph.nodes.items()
            if node.type == "practice"
        }

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        """쿼리와 유사한 문서를 PathRAG hybrid 방식으로 검색.

        파이프라인:
        1. Dense retrieval: 초기 후보 dense_k = k * 2개 검색
        2. Seed 노드: 초기 후보와 연결된 concept 노드 추출
        3. PathRAG BFS: Seed에서 시작해 경로 탐색 (max_hops=2, threshold=0.1)
        4. Score 결합: RRF(dense) + path_score * 0.5
        5. Top-k 반환

        Args:
            q: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            상위 k개 유사 문서
        """
        if not self.docs or self.dense.faiss_index is None:
            return []

        # Stage 1: Dense retrieval로 초기 후보 검색
        dense_k = k * 2
        dense_candidates = self.dense.search(q, dense_k)

        if not dense_candidates:
            return []

        # RRF score 계산 (rank-based fusion)
        dense_scores = {}
        for rank, doc in enumerate(dense_candidates):
            # RRF score: 1 / (rank + 60)
            rrf_score = 1.0 / (rank + 60)
            dense_scores[doc.id] = rrf_score

        # Stage 2: Seed 노드 추출 (초기 후보와 연결된 concept 노드)
        seed_nodes = set()
        for doc in dense_candidates:
            doc_id = doc.id
            # 문서 노드의 이웃 concept 노드들 추출
            neighbors = self.graph.get_neighbors(doc_id)
            for neighbor_id in neighbors:
                neighbor_node = self.graph.get_node(neighbor_id)
                if neighbor_node and neighbor_node.type != "practice":
                    # Concept 노드만 seed로 사용
                    seed_nodes.add(neighbor_id)

        # Seed가 없으면 dense 결과만 반환
        if not seed_nodes:
            return dense_candidates[:k]

        # Stage 3: PathRAG BFS 경로 탐색
        # 파라미터:
        # - max_hops=2: 2-hop 이내 탐색
        # - threshold=0.1: 최소 가중치 임계값 (낮은 값 = 더 많은 후보)
        # - alpha=0.8: 홉당 감쇠 계수
        # - use_soft_normalization=True: sqrt(degree) 정규화로 작은 그래프에서 더 나은 커버리지
        # - prune_during_exploration=False: 탐색 중 가지치기 비활성화 (더 많은 후보)
        path_scores = bfs_weighted_paths(
            adjacency=self.graph.adjacency,
            reverse_adjacency=self.graph.reverse_adjacency,
            sources=list(seed_nodes),
            practice_node_ids=self.practice_node_ids,
            max_hops=2,
            threshold=0.1,
            alpha=0.8,
            use_soft_normalization=True,
            prune_during_exploration=False,
        )

        # Stage 4: Score 결합 (RRF dense + path score * 0.5)
        combined_scores = {}

        # Dense 후보들 결합
        for doc_id, rrf_score in dense_scores.items():
            path_score = path_scores.get(doc_id, 0.0)
            # 결합 가중치: RRF + path * 0.5
            combined_scores[doc_id] = rrf_score + path_score * 0.5

        # PathRAG로만 발견된 문서들 추가
        for doc_id, path_score in path_scores.items():
            if doc_id not in combined_scores:
                # RRF 없이 path score만 사용 (가중치 0.5)
                combined_scores[doc_id] = path_score * 0.5

        # 점수 기준 정렬 후 상위 k개 선택
        sorted_doc_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )[:k]

        # SourceDoc 객체 복원
        doc_map = {d.id: d for d in self.docs}
        result_docs = []
        for doc_id in sorted_doc_ids:
            if doc_id in doc_map:
                result_docs.append(doc_map[doc_id])

        return result_docs

    def __repr__(self) -> str:
        return (
            f"PathRAGHybridRetriever("
            f"docs={len(self.docs)}, "
            f"graph_nodes={len(self.graph.nodes)}, "
            f"graph_edges={len(self.graph.edges)})"
        )
