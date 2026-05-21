"""Structured output for ``recall_check_node`` (prompt: ``prompts/recall_check.py``)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator


_FACT_KEY_RE = re.compile(r"^fact([1-9]\d*)$")


def _fact_key_index(key: str) -> int:
    match = _FACT_KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid fact key {key!r}; expected fact1, fact2, ...")
    return int(match.group(1))


def validate_fact_keys(keys: list[str]) -> None:
    """Enforce contiguous ``fact1``...``factN`` keys for deterministic routing."""

    indexes = [_fact_key_index(key) for key in keys]
    expected = list(range(1, len(indexes) + 1))
    if sorted(indexes) != expected:
        raise ValueError("Fact keys must be contiguous: fact1, fact2, ...")


def validate_fact_key_names(keys: list[str]) -> None:
    """Enforce ``factN`` key shape when a subset of facts is returned."""

    for key in keys:
        _fact_key_index(key)


class RequiredFactsResult(BaseModel):
    """Atomic information needs required to answer the normalized query."""

    facts: list["RequiredFact"] = Field(
        description=(
            "Ordered fact slots. Each item has fact_key (fact1, fact2, ...) and fact "
            "(an atomic information need, not the answer value)."
        ),
    )

    @model_validator(mode="after")
    def validate_facts(self) -> "RequiredFactsResult":
        if not self.facts:
            raise ValueError("facts must contain at least one fact")
        validate_fact_keys([item.fact_key for item in self.facts])
        return self


class RequiredFact(BaseModel):
    """One required fact slot from decomposition."""

    fact_key: str = Field(description="Contiguous key: fact1, fact2, ...")
    fact: str = Field(description="Atomic information need, not an answer value.")

    @field_validator("fact_key")
    @classmethod
    def validate_fact_key(cls, value: str) -> str:
        _fact_key_index(value)
        return value

    @field_validator("fact")
    @classmethod
    def validate_fact_text(cls, value: str) -> str:
        fact = value.strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        return fact


class FactVerification(BaseModel):
    """Evidence status for one required fact."""

    fact_key: str = Field(description="Fact key being verified: fact1, fact2, ...")
    fact: str = Field(description="Exact required fact text for this slot.")
    evidence_available: bool = Field(
        description="True when retrieved passages alone support this fact.",
    )
    evidence_documents: list[str] = Field(
        default_factory=list,
        description=(
            "Supporting excerpts copied from retrieved documents. Empty when "
            "evidence_available is false."
        ),
    )

    @field_validator("fact_key")
    @classmethod
    def validate_fact_key(cls, value: str) -> str:
        _fact_key_index(value)
        return value

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "FactVerification":
        fact = self.fact.strip()
        evidence_documents = [
            excerpt.strip() for excerpt in self.evidence_documents if excerpt and excerpt.strip()
        ]
        if not fact:
            raise ValueError("fact must be a non-empty string")
        if self.evidence_available and not evidence_documents:
            raise ValueError(
                "evidence_documents must be non-empty when evidence_available is true"
            )
        if not self.evidence_available and evidence_documents:
            raise ValueError(
                "evidence_documents must be empty when evidence_available is false"
            )
        return self.model_copy(update={"fact": fact, "evidence_documents": evidence_documents})


class RecallVerifyResult(BaseModel):
    """Per-fact sufficient-context verification against retrieved passages."""

    verifications: list[FactVerification] = Field(
        description="Verification result for every required fact.",
    )

    @model_validator(mode="after")
    def validate_verification_keys(self) -> "RecallVerifyResult":
        if not self.verifications:
            raise ValueError("verifications must contain at least one fact")
        validate_fact_keys([item.fact_key for item in self.verifications])
        return self
