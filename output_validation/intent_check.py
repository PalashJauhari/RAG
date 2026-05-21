"""Structured output for ``intent_check_node`` (prompt: ``prompts/intent_check.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from output_validation.recall_check import _fact_key_index, validate_fact_key_names


class FactIntentCheck(BaseModel):
    """Intent alignment for one unsupported fact."""

    fact_key: str = Field(description="Unsupported fact key: fact1, fact2, ...")
    fact: str = Field(description="Exact unsupported fact text for this slot.")
    intent_aligned: bool = Field(
        description="True when active_retrieval_queries target this fact's intent.",
    )
    intent_mismatch_details: str = Field(
        default="",
        description="Required when intent_aligned is false; empty when aligned.",
    )

    @field_validator("fact_key")
    @classmethod
    def validate_fact_key(cls, value: str) -> str:
        _fact_key_index(value)
        return value

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
    def validate_fact_intent_keys(self) -> "IntentCheckResult":
        if not self.fact_intents:
            raise ValueError("fact_intents must contain at least one fact")
        validate_fact_key_names([item.fact_key for item in self.fact_intents])
        return self
