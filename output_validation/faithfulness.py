"""Structured output for ``faithfulness_node`` (prompt: ``prompts/faithfulness.py``)."""

from pydantic import BaseModel, Field


class FaithfulnessResult(BaseModel):
    """Whether the answer is grounded in the cited catalog passages only."""

    passed: bool = Field(
        description="True when the answer is fully supported by the cited passages.",
    )
    reason: str = Field(
        description="Brief rationale for pass or fail (used as retry feedback when false).",
    )
