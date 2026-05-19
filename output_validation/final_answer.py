"""Structured output for ``answer_node`` and ``partial_answer_node`` (prompts: ``final_answer``, ``partial_answer``).

Parsed by ``api.main.get_api_response`` from ``AIMessage`` JSON in checkpointed ``messages``.
"""

from typing import Literal

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    answer: str = Field(description="Final answer grounded in retrieved documents.")
    sources: list[str] = Field(default_factory=list, description="Source labels or ids.")
    confidence: Literal["high", "medium", "low"] = Field(default="low")

