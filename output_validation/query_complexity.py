"""Structured output for ``query_complexity_node`` (prompt: ``prompts/query_complexity.py``).

The ``complexity`` label drives :meth:`~graph.graph.RetrievalGraph.route_after_complexity`.
``retrieval_strategy`` seeds the first retrieval pass for the turn.
"""

from typing import Literal

from pydantic import BaseModel, Field

from output_validation.retrieval_strategy import RetrievalStrategy


QueryComplexity = Literal[
    "simple_query",
    "comparison_query",
    "multihop_query",
    "procedural_query",
    "ambiguous_query",
    "exploratory_query",
]


class QueryComplexityResult(BaseModel):
    """Classification used to route the normalized query through the graph."""

    complexity: QueryComplexity = Field(
        description="Exact query complexity label used for graph routing.",
        examples=["comparison_query"],
    )
    retrieval_strategy: RetrievalStrategy = Field(
        description=(
            "Initial retrieval tier for this turn: fast_retrieval (dense+optional MMR only); "
            "fast_bm25_retrieval (dense + BM25 + RRF); keyword (BM25-only); "
            "fast_bm25_late_interaction_retrieval (hybrid + ColBERT re-rank when configured)."
        ),
        examples=["fast_bm25_retrieval"],
    )
    explanation: str = Field(
        description=(
            "Brief observable explanation covering routing label and why this retrieval_strategy fits."
        ),
        examples=["The user asks to compare two policy tiers; use hybrid lexical+dense retrieval."],
    )
