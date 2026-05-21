"""Structured output for ``recall_check_node`` (prompt: ``prompts/recall_check.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


def _strip_facts(facts: list[str]) -> list[str]:
    return [item.strip() for item in facts if item and item.strip()]


class RequiredFactsResult(BaseModel):
    """Atomic facts required to answer the normalized query."""

    required_facts: list[str] = Field(
        description="One atomic fact per string needed to answer faithfully.",
        min_length=1,
    )


class RecallCheckResult(BaseModel):
    """Whether retrieved passages support all required facts."""

    recall_sufficient: bool = Field(
        description="True when every required fact is supported by retrieved passages alone.",
    )
    missing_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Facts not supported by passages. One atomic fact per string. "
            "Empty when recall_sufficient is true."
        ),
    )

    @model_validator(mode="after")
    def validate_recall_fields(self) -> "RecallCheckResult":
        missing = _strip_facts(self.missing_facts)
        if self.recall_sufficient:
            if missing:
                raise ValueError("missing_facts must be empty when recall_sufficient is true")
            return self.model_copy(update={"missing_facts": []})
        if not missing:
            raise ValueError(
                "missing_facts must contain at least one non-empty string when recall_sufficient is false"
            )
        return self.model_copy(update={"missing_facts": missing})
