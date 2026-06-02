"""Structured output for ``answer_node``, ``partial_answer_node``, and ``error_answer_node``.

Used by prompts ``final_answer`` and ``partial_answer`` plus deterministic fallback in
``error_answer_node``. Parsed by ``api.main.get_api_response`` from ``AIMessage`` JSON
in checkpointed ``messages``.
"""

from typing import Literal

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    answer: str = Field(description="Final answer grounded in retrieved documents.")
    sources: list[str] = Field(default_factory=list, description="Source labels or ids.")
    confidence: Literal["high", "medium", "low"] = Field(default="low")

