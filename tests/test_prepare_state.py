"""prepare_state_for_next_question clears catalog and citation scratch."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from graph.graph import prepare_state_for_next_question


def test_prepare_state_resets_catalog_and_retry_counters() -> None:
    patch = prepare_state_for_next_question("hello")
    assert isinstance(patch["messages"][0], HumanMessage)
    assert patch["messages"][0].content == "hello"
    assert patch["document_catalog"] == {}
    assert patch["cited_document_ids"] == []
    assert patch["cited_id_retry_count"] == 0
    assert patch["answer_retry_count"] == 0
    assert patch["answer_mode"] == ""
    assert patch["final_sources"] == []
    assert patch["faithfulness_feedback"] == ""
    assert patch["cited_id_feedback"] == ""
    assert "retrieved_documents" not in patch
    assert "retrieved_point_ids" not in patch
