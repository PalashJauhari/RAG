"""Shared literals for per-request retrieval strategy (graph + structured outputs)."""

from typing import Literal

RetrievalStrategy = Literal[
    "fast_retrieval",
    "fast_bm25_retrieval",
    "keyword",
    "fast_bm25_late_interaction_retrieval",
]
