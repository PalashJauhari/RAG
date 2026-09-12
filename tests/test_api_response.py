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
        "final_sources": ["id-1"],
    }
    response = get_api_response("sess", result)
    assert response["answer"] == "done"
    assert response["sources"] == ["id-1"]
    assert response["cited_document_ids"] == ["id-1"]
    assert response["document_catalog"] == {"id-1": {"source": "hotpotqa"}}
    assert response["cited_ui"] == [{"source": "hotpotqa"}]
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
    assert response["cited_ui"] == []
    assert response["document_catalog"] == {}


def test_get_api_response_cited_ui_media_and_hotpot_without_media() -> None:
    message = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "done",
            "cited_document_ids": ["keep", "hotpot", "missing"],
            "confidence": "high",
            "sources": [],
        },
    )
    result = {
        "messages": [message],
        "document_catalog": {
            "keep": {
                "text": "Relevant table data:\n<table>x</table>",
                "source": "https://arxiv.org/abs/1",
                "page_number": 9,
                "images_base64": ["YmFzZTY0"],
                "table_html": ["<table>x</table>"],
            },
            "hotpot": {"text": "wiki", "source": "hotpotqa", "images_base64": None},
            "other": {"text": "uncited", "images_base64": ["nope"]},
        },
        "cited_document_ids": ["keep", "hotpot", "missing"],
        "final_sources": ["keep", "hotpot"],
    }
    response = get_api_response("sess", result)
    assert response["answer"] == "done"
    assert "text" not in response["document_catalog"]["keep"]
    assert response["document_catalog"]["keep"]["images_base64"] == ["YmFzZTY0"]
    assert "other" not in response["document_catalog"]
    assert response["cited_ui"] == [
        {
            "source": "https://arxiv.org/abs/1",
            "page_number": 9,
            "images_base64": ["YmFzZTY0"],
            "table_html": ["<table>x</table>"],
        },
        {"source": "hotpotqa"},
    ]
