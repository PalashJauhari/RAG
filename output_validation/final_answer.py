"""Structured output for ``answer_node``, ``partial_answer_node``, and ``error_answer_node``.

Used by prompts ``final_answer`` and ``partial_answer`` plus deterministic fallback in
``error_answer_node``. Parsed by ``api.main.get_api_response`` from ``AIMessage`` JSON
in checkpointed ``messages``.

``cited_document_ids`` are chosen by the LLM from ``document_catalog`` keys.
``sources`` is filled by code after faithfulness (LLM must leave it empty).
"""

from typing import Literal

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    answer: str = Field(description="Final answer grounded in retrieved documents.")
    cited_document_ids: list[str] = Field(
        default_factory=list,
        description="Document catalog point ids used to ground the answer.",
    )
    confidence: Literal["high", "medium", "low"] = Field(default="low")
    sources: list[str] = Field(
        default_factory=list,
        description="Source labels filled by code from document_catalog; LLM must return [].",
    )
