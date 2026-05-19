"""Structured output for ``query_rewriter_node`` (prompt: ``prompts/query_rewriter.py``)."""

from pydantic import BaseModel, Field


class QueryRewriteResult(BaseModel):
    """Best-effort rewrite for ambiguous queries when no human clarification is taken yet."""

    rewritten_query: str = Field(
        description="A cautious retrieval-ready rewrite of the ambiguous normalized query.",
        examples=["refund policy rules for the user's referenced product or tier"],
    )
    rewrite_explanation: str = Field(
        description="Brief explanation of how ambiguity was handled without inventing facts.",
        examples=[
            "Kept the ambiguous tier reference explicit instead of assuming one tier.",
        ],
    )
