"""Structured output for HotpotQA context enrichment at prepare time."""

from pydantic import BaseModel, Field, field_validator, model_validator


class ContextFact(BaseModel):
    """One checkable information need and a searchable query for it."""

    fact: str = Field(description="Atomic information need phrased as what must be established.")
    fact_question: str = Field(description="Self-contained retrieval query for that need.")

    @field_validator("fact", "fact_question")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned


class ContextEnrichmentResult(BaseModel):
    """LLM-generated metadata for one HotpotQA context passage."""

    predicted_title: str = Field(description="Suggested title for the passage.")
    summary: str = Field(description="Two-line summary of the passage.")
    keywords: list[str] = Field(
        default_factory=list,
        description="Named entities and other high-signal retrieval terms.",
    )
    facts: list[ContextFact] = Field(
        default_factory=list,
        description="Checkable needs and paired search queries.",
    )

    @field_validator("predicted_title", "summary")
    @classmethod
    def strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def normalize_keywords(self) -> "ContextEnrichmentResult":
        seen: set[str] = set()
        keywords: list[str] = []
        for item in self.keywords:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                keywords.append(cleaned)
                seen.add(cleaned)
        self.keywords = keywords
        return self
