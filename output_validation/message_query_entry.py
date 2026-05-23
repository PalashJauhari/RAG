"""Append-only audit row for ``message_query`` state.

Each graph node may emit one or more rows describing what happened in that step (queries,
retrieval, evaluation, gap_fill, strategy_upgrade).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MessageQueryEntry(BaseModel):
    """One trace row appended to ``message_query`` during a user turn."""

    node: str = Field(description="LangGraph node id that produced this row.")
    kind: str = Field(
        description=(
            "Trace category such as normalisation, fact_decomposition, complexity, "
            "retrieval, recall_check, gap_fill, or strategy_upgrade."
        ),
    )
    payload: dict = Field(default_factory=dict, description="JSON-safe detail for this step.")
    notes: str | None = Field(default=None, description="Optional human-readable summary.")
