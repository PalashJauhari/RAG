"""Structured output for ``gap_fill_node`` (prompt: ``prompts/gap_fill.py``).

``fact_queries`` replace ``active_retrieval_queries`` on insufficient recall retries.
Tier selection is deterministic in the graph.
"""

from pydantic import BaseModel, Field, model_validator

from output_validation.fact_decomposition import validate_unique_fact_texts


def ensure_three_search_queries(fact: str, queries: list[str]) -> list[str]:
    """Return exactly three unique non-empty queries, padding from ``fact`` when the LLM under-fills."""

    fact = fact.strip()
    seen: set[str] = set()
    normalized: list[str] = []
    for query in queries:
        cleaned = query.strip()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)

    if len(normalized) >= 3:
        return normalized[:3]

    fallbacks = [
        fact,
        f"{fact} documents passages" if fact else "",
        f"{fact} keyword search" if fact else "",
    ]
    for candidate in fallbacks:
        if len(normalized) >= 3:
            break
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)

    suffix = 1
    while len(normalized) < 3:
        base = normalized[0] if normalized else fact or "retrieval query"
        candidate = f"{base} alternate phrasing {suffix}"
        suffix += 1
        if candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)

    return normalized[:3]


class FactSearchQueries(BaseModel):
    """Three retrieval queries for one unsupported fact."""

    fact: str = Field(description="Exact unsupported fact text.")

    search_queries: list[str] = Field(
        description="Exactly three focused, self-contained retrieval queries.",
    )

    @model_validator(mode="after")
    def validate_fact_and_queries(self) -> "FactSearchQueries":
        fact = self.fact.strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        self.fact = fact
        self.search_queries = ensure_three_search_queries(fact, self.search_queries)
        return self


class GapFillResult(BaseModel):
    """Missing-evidence queries generated after insufficient recall."""

    fact_queries: list[FactSearchQueries] = Field(
        description="Exactly three search queries for each unsupported fact.",
    )
    gap_fill_explanation: str = Field(
        description="Brief explanation of how the missing queries address the recall gap.",
        examples=[
            "Generated one query for each missing tier-specific refund detail.",
        ],
    )

    @model_validator(mode="after")
    def validate_fact_queries(self) -> "GapFillResult":
        if not self.fact_queries:
            raise ValueError("fact_queries must contain at least one fact")
        validate_unique_fact_texts([item.fact for item in self.fact_queries])
        return self
