"""Structured output for ``recall_check_node`` (prompt: ``prompts/recall_check.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from output_validation.fact_decomposition import validate_unique_fact_texts


class VerifiedFact(BaseModel):
    """Evidence status for one fact from the unified facts list."""

    fact_id: int | None = Field(
        default=None,
        description="Stable fact id from fact_decomposition; preserved when returned.",
    )
    fact: str = Field(description="Exact fact text from the unified facts list.")
    verification_status: bool = Field(
        description="True when retrieved passages alone support this fact.",
    )
    evidence_document_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Catalog point ids that support this fact. Empty when "
            "verification_status is false."
        ),
    )

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "VerifiedFact":
        fact = self.fact.strip()
        evidence_document_ids = [
            point_id.strip()
            for point_id in self.evidence_document_ids
            if point_id and str(point_id).strip()
        ]
        if not fact:
            raise ValueError("fact must be a non-empty string")
        if self.verification_status and not evidence_document_ids:
            raise ValueError(
                "evidence_document_ids must be non-empty when verification_status is true"
            )
        if not self.verification_status and evidence_document_ids:
            raise ValueError(
                "evidence_document_ids must be empty when verification_status is false"
            )
        return self.model_copy(
            update={
                "fact": fact,
                "evidence_document_ids": evidence_document_ids,
            }
        )


class RecallVerifyResult(BaseModel):
    """Batch verification result for one recall-check LLM call (all facts)."""

    facts: list[VerifiedFact] = Field(
        description="Verification result for every fact in the unified facts list.",
    )

    @model_validator(mode="after")
    def validate_verified_facts(self) -> "RecallVerifyResult":
        if not self.facts:
            raise ValueError("facts must contain at least one fact")
        validate_unique_fact_texts([item.fact for item in self.facts])
        return self
