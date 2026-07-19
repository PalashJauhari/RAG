"""Benchmark helpers stay aligned with document_catalog + faithfulness state."""

from __future__ import annotations

from graph.graph import build_node_ai_message
from benchmarking.hotpotqa.run_evaluation import graph_run_response, is_partial_answer


def test_graph_run_response_uses_final_sources_and_catalog() -> None:
    message = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "done",
            "cited_document_ids": ["id-1"],
            "confidence": "high",
            "sources": [],
        },
    )
    state = {
        "messages": [message],
        "document_catalog": {
            "id-1": {"text": "passage", "source": "hotpotqa", "score": 0.9},
        },
        "cited_document_ids": ["id-1"],
        "final_sources": ["hotpotqa"],
        "faithfulness_ok": True,
        "answer_mode": "full",
    }
    response = graph_run_response("sess", state)
    assert response["answer"] == "done"
    assert response["sources"] == ["hotpotqa"]
    assert response["cited_document_ids"] == ["id-1"]
    assert response["document_catalog"]["id-1"]["text"] == "passage"
    assert response["faithfulness_ok"] is True
    assert response["answer_mode"] == "full"


def test_is_partial_answer_uses_answer_mode() -> None:
    assert is_partial_answer({"answer_mode": "partial", "messages": []}) is True
    assert is_partial_answer({"answer_mode": "full", "messages": []}) is False


def test_is_partial_answer_falls_back_to_message_name() -> None:
    message = build_node_ai_message(
        node_name="partial_answer_node",
        payload={
            "answer": "partial",
            "cited_document_ids": [],
            "confidence": "low",
            "sources": [],
        },
    )
    assert is_partial_answer({"answer_mode": "", "messages": [message]}) is True
