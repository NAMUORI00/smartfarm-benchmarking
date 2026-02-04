"""Ours (full) retriever baseline.

Dense + Sparse + Tri-Graph via weighted RRF fusion (current production retriever).
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()

from core.Models.Schemas import SourceDoc
from core.Services.Retrieval.Base import BaseRetriever


@runtime_checkable
class FusionRetrieverProtocol(Protocol):
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...


class OursFullRetriever(BaseRetriever):
    """Full (production) fusion retriever baseline."""

    def __init__(self, fusion: FusionRetrieverProtocol):
        self.fusion = fusion

    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        return self.fusion.search(q, k)

    def __repr__(self) -> str:
        return "OursFullRetriever()"

