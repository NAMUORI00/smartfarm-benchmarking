"""Tri-Graph-only retriever baseline.

Graph-based multi-hop retrieval channel only.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Base import BaseRetriever


@runtime_checkable
class TriGraphRetrieverProtocol(Protocol):
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


class TriGraphOnlyRetriever(BaseRetriever):
    """Tri-Graph-only baseline retriever."""

    def __init__(self, trigraph: TriGraphRetrieverProtocol):
        self.trigraph = trigraph

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        return self.trigraph.search(q, k)

    def __repr__(self) -> str:
        return "TriGraphOnlyRetriever()"

