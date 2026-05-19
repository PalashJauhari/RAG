"""Structured output for ``gap_fill_node`` (prompt: ``prompts/gap_fill.py``).

``missing_queries`` replace ``active_retrieval_queries`` on insufficient_recall retries.
Optional ``next_retrieval_strategy`` may bump the retrieval tier when lexical recall is weak.
"""

from pydantic import BaseModel, Field

from output_validation.retrieval_strategy import RetrievalStrategy


class GapFillResult(BaseModel):
    """Missing-evidence queries generated after insufficient recall."""

    missing_queries: list[str] = Field(
        description="Focused retrieval queries targeting the missing evidence.",
        examples=[["consumer refund policy cancellation window", "enterprise refund policy exceptions"]],
    )
    gap_fill_explanation: str = Field(
        description="Brief explanation of how the missing queries address the recall gap.",
        examples=[
            "Generated one query for each missing tier-specific refund detail.",
        ],
    )
    next_retrieval_strategy: RetrievalStrategy | None = Field(
        default=None,
        description=(
            "Optional escalation of retrieval tier after insufficient recall (e.g. add BM25). "
            "Omit to keep the current retrieval_strategy and only refresh queries."
        ),
        examples=["fast_bm25_retrieval"],
    )
