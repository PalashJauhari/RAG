"""Structured output for ``query_splitter_node`` (prompt: ``prompts/query_splitter.py``)."""

from pydantic import BaseModel, Field


class QuerySplitResult(BaseModel):
    queries: list[str] = Field(description="Focused retrieval queries.")

