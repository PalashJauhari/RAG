"""FaithfulnessResult schema."""

from __future__ import annotations

from output_validation.faithfulness import FaithfulnessResult


def test_faithfulness_result_fields() -> None:
    result = FaithfulnessResult(passed=True, reason="supported")
    assert result.passed is True
    assert result.reason == "supported"
