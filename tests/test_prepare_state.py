"""prepare_state_for_next_question clears catalog and faithfulness scratch."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from graph.graph import prepare_state_for_next_question


def test_prepare_state_resets_catalog_and_faithfulness_scratch() -> None:
    patch = prepare_state_for_next_question("hello")
    assert isinstance(patch["messages"][0], HumanMessage)
    assert patch["messages"][0].content == "hello"
    assert patch["user_question"] == "hello"
    assert patch["document_catalog"] == {}
    assert patch["cited_document_ids"] == []
    assert patch["answer_text"] == ""
    assert patch["strategies_used"] == []
    assert patch["faithfulness_retry_count"] == 0
    assert patch["faithfulness_ok"] is False
    assert patch["faithfulness_forced_pass"] is False
    assert patch["faithfulness_feedback"] == ""
    assert patch["answer_mode"] == ""
    assert patch["final_sources"] == []
    assert "cited_id_check_retry_count" not in patch
    assert "cited_id_check_feedback" not in patch
    assert "faithfulness_answer_retry_count" not in patch
    assert "retrieved_documents" not in patch
