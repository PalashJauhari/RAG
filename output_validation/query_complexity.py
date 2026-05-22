"""Structured output for ``query_complexity_node`` (prompt: ``prompts/query_complexity.py``).

The ``complexity`` label drives :meth:`~graph.graph.RetrievalGraph.route_after_complexity`.
"""

from typing import Literal

from pydantic import BaseModel, Field

QueryComplexity = Literal["simple_query", "needs_split"]


class QueryComplexityResult(BaseModel):
    """Classification used to route the normalized query through the graph."""

    complexity: QueryComplexity = Field(
        description="Exact query complexity label used for graph routing.",
        examples=["needs_split"],
    )
    explanation: str = Field(
        description="Brief observable explanation covering only the routing label.",
        examples=["The required facts cover multiple entities, so route to query_splitter."],
    )
