"""Structured output for ``intent_correction_rewriter_node`` (prompt: ``prompts/intent_correction_rewriter.py``).

``fact_queries`` replace ``active_retrieval_queries`` after intent_mismatch.
"""

from pydantic import BaseModel, Field, model_validator

from output_validation.gap_fill import FactSearchQueries
from output_validation.recall_check import validate_fact_key_names


class IntentCorrectionRewriteResult(BaseModel):
    """Corrected per-fact retrieval queries after intent_check detects misalignment."""

    fact_queries: list[FactSearchQueries] = Field(
        description="Exactly three corrected search queries for each misaligned fact.",
    )
    correction_explanation: str = Field(
        description="Brief explanation of what intent drift was corrected.",
        examples=[
            "Removed wording that pulled retrieval toward general pricing documents.",
        ],
    )

    @model_validator(mode="after")
    def validate_fact_query_keys(self) -> "IntentCorrectionRewriteResult":
        if not self.fact_queries:
            raise ValueError("fact_queries must contain at least one fact")
        validate_fact_key_names([item.fact_key for item in self.fact_queries])
        return self
