"""Structured output for ``query_normalisation_node`` (prompt: ``prompts/query_normalisation.py``)."""

from pydantic import BaseModel, Field


class QueryNormalisationResult(BaseModel):
    """Standalone user query produced before routing and retrieval."""

    normalized_query: str = Field(
        description=(
            "A clear, standalone version of the latest user query, rewritten with useful "
            "conversation context and without answering the query."
        ),
        examples=["Compare the Enterprise refund policy with the Consumer refund policy."],
    )
    normalization_explanation: str = Field(
        description="Brief observable explanation of what context or wording was normalized.",
        examples=["Resolved the follow-up reference to the refund policy tiers."],
    )
