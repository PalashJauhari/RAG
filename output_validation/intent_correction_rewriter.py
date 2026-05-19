"""Structured output for ``intent_correction_rewriter_node`` (prompt: ``prompts/intent_correction_rewriter.py``).

``corrected_queries`` replace ``active_retrieval_queries`` after intent_mismatch.
Optional ``next_retrieval_strategy`` may bump tier when appropriate.
"""

from pydantic import BaseModel, Field

from output_validation.retrieval_strategy import RetrievalStrategy


class IntentCorrectionRewriteResult(BaseModel):
    """Corrected retrieval queries after the evaluator detects intent mismatch."""

    corrected_queries: list[str] = Field(
        description="Retrieval queries rewritten to better match the intended user request.",
        examples=[["enterprise refund policy", "consumer refund policy"]],
    )
    correction_explanation: str = Field(
        description="Brief explanation of what intent drift was corrected.",
        examples=[
            "Removed wording that pulled retrieval toward general pricing documents.",
        ],
    )
    next_retrieval_strategy: RetrievalStrategy | None = Field(
        default=None,
        description=(
            "Optional retrieval tier change after intent drift (e.g. keyword when embeddings "
            "keep missing exact entities). Omit to keep the current retrieval_strategy."
        ),
        examples=["keyword"],
    )
