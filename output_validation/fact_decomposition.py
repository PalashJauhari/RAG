"""Structured output for ``fact_decomposition_node``.

Creates stable, ordered information needs from the normalized query before retrieval.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_unique_fact_texts(facts: list[str]) -> None:
    """Reject duplicate fact strings so downstream routing can match by exact text."""

    if len(set(facts)) != len(facts):
        raise ValueError("facts must be unique")


class RequiredFactsResult(BaseModel):
    """Atomic information needs required to answer the normalized query."""

    facts: list["RequiredFact"] = Field(
        description=(
            "Ordered fact objects. Each fact is an atomic information need, "
            "not the answer value."
        ),
    )

    @model_validator(mode="after")
    def validate_facts(self) -> "RequiredFactsResult":
        if not self.facts:
            raise ValueError("facts must contain at least one fact")
        validate_unique_fact_texts([item.fact for item in self.facts])
        return self


class RequiredFact(BaseModel):
    """One required information need from decomposition."""

    fact: str = Field(description="Atomic information need, not an answer value.")

    @field_validator("fact")
    @classmethod
    def validate_fact_text(cls, value: str) -> str:
        fact = value.strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        return fact
