from typing import Literal

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    answer: str = Field(description="Final answer grounded in retrieved documents.")
    sources: list[str] = Field(default_factory=list, description="Source labels or ids.")
    confidence: Literal["high", "medium", "low"] = Field(default="low")

