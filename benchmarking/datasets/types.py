from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

CorpusDoc = Tuple[str, str]


@dataclass(frozen=True)
class EvalSample:
    qid: str
    question: str
    relevant_doc_ids: Set[str]
    relevance_scores: Dict[str, float]
    gold_answer: str = ""

