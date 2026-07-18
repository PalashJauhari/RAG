"""Query normalisation prompt context must omit Latest User Query block."""

from __future__ import annotations

from prompts import query_normalisation


def test_query_normalisation_prompt_has_no_latest_user_query_section() -> None:
    prompt = query_normalisation.SYSTEM_PROMPT
    assert "## Conversation Summary" in prompt
    assert "## Recent Messages" in prompt
    assert "## Latest User Query" not in prompt
