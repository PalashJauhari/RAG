"""FinalAnswer pydantic schema fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from output_validation.final_answer import FinalAnswer


def test_final_answer_requires_cited_ids_field() -> None:
    result = FinalAnswer(
        answer="ok",
        cited_document_ids=["a"],
        confidence="high",
        sources=[],
    )
    assert result.cited_document_ids == ["a"]
    assert result.sources == []


def test_final_answer_defaults() -> None:
    result = FinalAnswer(answer="x")
    assert result.cited_document_ids == []
    assert result.sources == []
    assert result.confidence == "low"


def test_final_answer_rejects_bad_confidence() -> None:
    with pytest.raises(ValidationError):
        FinalAnswer(answer="x", confidence="nope")
