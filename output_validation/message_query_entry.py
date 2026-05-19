"""Structured audit row appended into LangGraph ``message_query`` (operator.add reducer)."""

from typing import Any

from pydantic import BaseModel, Field


class MessageQueryEntry(BaseModel):
    """One trace row emitted by a graph node."""

    node: str = Field(description="LangGraph node id that produced this entry.")
    kind: str = Field(
        description=(
            "Trace category, e.g. normalisation, complexity, query_prep (splitter/expansion/rewriter), "
            "retrieval, evaluation, gap_fill, intent_correction."
        ),
    )
    payload: dict[str, Any] = Field(default_factory=dict, description="JSON-safe detail blob.")
    notes: str | None = Field(default=None, description="Optional short human summary.")
