"""Structured output for ``intent_check_node`` (prompt: ``prompts/intent_check.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from output_validation.fact_decomposition import validate_unique_fact_texts


class FactIntentCheck(BaseModel):
    """Intent alignment for one unsupported fact."""

    fact: str = Field(description="Exact unsupported fact text.")
    intent_aligned: bool = Field(
        description="True when active_retrieval_queries target this fact's intent.",
    )
    intent_mismatch_details: str = Field(
        default="",
        description="Required when intent_aligned is false; empty when aligned.",
    )

    @model_validator(mode="after")
    def validate_intent_fields(self) -> "FactIntentCheck":
        fact = self.fact.strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        details = (self.intent_mismatch_details or "").strip()
        if not self.intent_aligned and not details:
            raise ValueError(
                "intent_mismatch_details must be non-empty when intent_aligned is false"
            )
        if self.intent_aligned and details:
            raise ValueError("intent_mismatch_details must be empty when intent_aligned is true")
        return self.model_copy(update={"fact": fact, "intent_mismatch_details": details})


class IntentCheckResult(BaseModel):
    """Per-fact intent alignment for unsupported recall facts."""

    fact_intents: list[FactIntentCheck] = Field(
        description="Intent checks for unsupported facts.",
    )

    @model_validator(mode="after")
    def validate_fact_intents(self) -> "IntentCheckResult":
        if not self.fact_intents:
            raise ValueError("fact_intents must contain at least one fact")
        validate_unique_fact_texts([item.fact for item in self.fact_intents])
        return self
