from __future__ import annotations

from .agxqa import load_agxqa
from .twowiki import load_twowiki_multihopqa
from .types import CorpusDoc, EvalSample

__all__ = [
    "CorpusDoc",
    "EvalSample",
    "load_agxqa",
    "load_twowiki_multihopqa",
]

