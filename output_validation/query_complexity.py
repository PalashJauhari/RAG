from typing import Literal

from pydantic import BaseModel, Field


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
    explanation: str = Field(
        description="Brief observable explanation for the selected complexity label.",
        examples=["The user asks to compare two policy tiers, so separate retrieval is useful."],
    )
