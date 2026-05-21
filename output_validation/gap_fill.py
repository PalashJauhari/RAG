"""Structured output for ``gap_fill_node`` (prompt: ``prompts/gap_fill.py``).

``fact_queries`` replace ``active_retrieval_queries`` on insufficient recall retries.
Tier selection is deterministic in the graph.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

from output_validation.recall_check import _fact_key_index, validate_fact_key_names


class FactSearchQueries(BaseModel):
    """Three retrieval queries for one unsupported fact."""

    fact_key: str = Field(description="Fact key for these queries: fact1, fact2, ...")
    fact: str = Field(description="Exact unsupported fact text for this slot.")
    @field_validator("fact_key")
    @classmethod
    def validate_fact_key(cls, value: str) -> str:
        _fact_key_index(value)
        return value

    search_queries: list[str] = Field(
        description="Exactly three focused, self-contained retrieval queries.",
    )

    @field_validator("search_queries")
    @classmethod
    def validate_three_queries(cls, value: list[str]) -> list[str]:
        queries = [query.strip() for query in value if query and query.strip()]
        if len(queries) != 3:
            raise ValueError("search_queries must contain exactly three non-empty queries")
        return queries

    @model_validator(mode="after")
    def validate_fact(self) -> "FactSearchQueries":
        fact = self.fact.strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        return self.model_copy(update={"fact": fact})


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
    def validate_fact_query_keys(self) -> "GapFillResult":
        if not self.fact_queries:
            raise ValueError("fact_queries must contain at least one fact")
        validate_fact_key_names([item.fact_key for item in self.fact_queries])
        return self
