"""Structured output for ``fact_gap_retrieval_node`` search-query conversion."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class FactGapQueryResult(BaseModel):
    """One BM25-friendly search query per uncovered fact (1:1 with ``missing_facts``)."""

    search_queries: list[str] = Field(
        description="Search queries aligned 1:1 with missing facts in the same order.",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_queries(self) -> "FactGapQueryResult":
        cleaned = [q.strip() for q in self.search_queries if q and q.strip()]
        if not cleaned:
            raise ValueError("search_queries must contain at least one non-empty string")
        object.__setattr__(self, "search_queries", cleaned)
        return self
