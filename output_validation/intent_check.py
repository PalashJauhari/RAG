"""Structured output for ``intent_check_node`` (prompt: ``prompts/intent_check.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class IntentCheckResult(BaseModel):
    """Whether active retrieval queries align with the normalized query intent."""

    intent_aligned: bool = Field(
        description="True when active_retrieval_queries match normalized query intent and entities.",
    )
    intent_mismatch_details: str = Field(
        default="",
        description="Required when intent_aligned is false: detailed explanation for the rewriter.",
    )

    @model_validator(mode="after")
    def validate_intent_fields(self) -> "IntentCheckResult":
        details = (self.intent_mismatch_details or "").strip()
        if not self.intent_aligned and not details:
            raise ValueError(
                "intent_mismatch_details must be non-empty when intent_aligned is false"
            )
        if self.intent_aligned and details:
            raise ValueError("intent_mismatch_details must be empty when intent_aligned is true")
        return self
