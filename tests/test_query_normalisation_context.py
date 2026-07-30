"""Query normalisation prompt documents User question + summary + messages."""

from __future__ import annotations

from prompts import query_normalisation


def test_query_normalisation_prompt_documents_user_question_block() -> None:
    prompt = query_normalisation.SYSTEM_PROMPT
    assert "## User question" in prompt
    assert "## Conversation Summary" in prompt
    assert "## Recent Messages" in prompt
