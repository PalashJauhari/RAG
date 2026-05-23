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
    verification_report: str = Field(
        description="Brief rationale explaining why the fact is or is not supported.",
    )
    evidence_documents: list[str] = Field(
        default_factory=list,
        description=(
            "Supporting excerpts copied from retrieved documents. Empty when "
            "verification_status is false."
        ),
    )

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "VerifiedFact":
        fact = self.fact.strip()
        report = self.verification_report.strip()
        evidence_documents = [
            excerpt.strip() for excerpt in self.evidence_documents if excerpt and excerpt.strip()
        ]
        if not fact:
            raise ValueError("fact must be a non-empty string")
        if not report:
            raise ValueError("verification_report must be a non-empty string")
        if self.verification_status and not evidence_documents:
            raise ValueError(
                "evidence_documents must be non-empty when verification_status is true"
            )
        if not self.verification_status and evidence_documents:
            raise ValueError(
                "evidence_documents must be empty when verification_status is false"
            )
        return self.model_copy(
            update={
                "fact": fact,
                "verification_report": report,
                "evidence_documents": evidence_documents,
            }
        )


class RecallVerifyResult(BaseModel):
    """Per-fact sufficient-context verification against retrieved passages."""

    facts: list[VerifiedFact] = Field(
        description="Verification result for every fact in the unified facts list.",
    )

    @model_validator(mode="after")
    def validate_verified_facts(self) -> "RecallVerifyResult":
        if not self.facts:
            raise ValueError("facts must contain at least one fact")
        validate_unique_fact_texts([item.fact for item in self.facts])
        return self
