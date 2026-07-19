"""API response shaping for document_catalog and sources."""

from __future__ import annotations

from graph.graph import build_node_ai_message
from api.main import get_api_response


def test_get_api_response_returns_document_catalog_and_final_sources() -> None:
    message = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "done",
            "cited_document_ids": ["id-1"],
            "confidence": "medium",
            "sources": [],
        },
    )
    result = {
        "messages": [message],
        "document_catalog": {
            "id-1": {"text": "passage", "source": "hotpotqa", "score": 0.9},
        },
        "cited_document_ids": ["id-1"],
        "final_sources": ["hotpotqa"],
    }
    response = get_api_response("sess", result)
    assert response["answer"] == "done"
    assert response["sources"] == ["hotpotqa"]
    assert response["cited_document_ids"] == ["id-1"]
    assert response["document_catalog"]["id-1"]["text"] == "passage"
    assert "retrieved_docs" not in response


def test_get_api_response_empty_final_sources_when_key_present() -> None:
    message = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "latest",
            "cited_document_ids": ["bad"],
            "confidence": "low",
            "sources": ["should-not-use"],
        },
    )
    result = {
        "messages": [message],
        "document_catalog": {},
        "cited_document_ids": ["bad"],
        "final_sources": [],
    }
    response = get_api_response("sess", result)
    assert response["sources"] == []
    assert response["answer"] == "latest"
