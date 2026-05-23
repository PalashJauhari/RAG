"""Structured output for ``gap_fill_node`` (prompt: ``prompts/gap_fill.py``).

Repairs unsupported facts in place on the unified ``facts`` list; the graph flattens
``search_queries`` into ``active_retrieval_queries`` for retrieval.
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


class GapFillFact(BaseModel):
    """Retrieval repair for one unsupported fact."""

    fact_id: int = Field(description="Stable fact id from the unified facts list.")
    fact: str = Field(description="Exact unsupported fact text.")
    search_queries: list[str] = Field(
        description="Exactly three focused, self-contained retrieval queries.",
    )
    gap_fill_explanation: str = Field(
        description="Brief explanation of how these queries address this fact's recall gap.",
    )

    @model_validator(mode="after")
    def validate_repair_fields(self) -> "GapFillFact":
        fact = self.fact.strip()
        explanation = self.gap_fill_explanation.strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        if not explanation:
            raise ValueError("gap_fill_explanation must be a non-empty string")
        return self.model_copy(
            update={
                "fact": fact,
                "gap_fill_explanation": explanation,
                "search_queries": ensure_three_search_queries(fact, self.search_queries),
            }
        )


class GapFillResult(BaseModel):
    """Per-fact repair rows for unsupported facts only."""

    facts: list[GapFillFact] = Field(
        description="One repair object for every unsupported fact and no extras.",
    )

    @model_validator(mode="after")
    def validate_facts(self) -> "GapFillResult":
        if not self.facts:
            raise ValueError("facts must contain at least one unsupported fact")
        validate_unique_fact_texts([item.fact for item in self.facts])
        fact_ids = [item.fact_id for item in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("fact_id values must be unique within gap-fill output")
        return self
