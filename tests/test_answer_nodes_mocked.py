"""Answer / validate / faithfulness nodes with mocked LLM (no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from config import settings as settings_module
from graph.graph import RetrievalGraph, build_node_ai_message
from output_validation.faithfulness import FaithfulnessResult
from output_validation.final_answer import FinalAnswer


class FakeStructuredClient:
    def __init__(self, parsed: Any) -> None:
        self.parsed = parsed
        self.last_messages: list[Any] | None = None

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.last_messages = messages
        raw = AIMessage(content="raw")
        return {"parsed": self.parsed, "raw": raw}


@pytest.mark.asyncio
async def test_answer_node_sets_mode_and_cited_ids(monkeypatch) -> None:
    parsed = FinalAnswer(
        answer="The answer",
        cited_document_ids=["doc-1"],
        confidence="high",
        sources=["should-be-cleared"],
    )
    fake = FakeStructuredClient(parsed)
    monkeypatch.setattr(
        "graph.graph.get_llm_client",
        lambda **kwargs: fake,
    )
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)

    graph = RetrievalGraph(InMemorySaver())
    state = {
        "normalized_query": "what?",
        "document_catalog": {
            "doc-1": {"text": "evidence", "source": "hotpotqa", "score": 0.9},
        },
        "cited_id_check_feedback": "",
        "faithfulness_feedback": "",
    }
    result = await graph.answer_node(state)
    assert result["answer_mode"] == "full"
    assert result["cited_document_ids"] == ["doc-1"]
    payload = result["messages"][0]
    assert payload.name == "answer_node"
    assert '"sources": []' in payload.content or '"sources":[]' in payload.content.replace(" ", "")


@pytest.mark.asyncio
async def test_answer_node_includes_ai_feedback_messages(monkeypatch) -> None:
    parsed = FinalAnswer(answer="retry", cited_document_ids=["doc-1"], confidence="low")
    fake = FakeStructuredClient(parsed)
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)

    graph = RetrievalGraph(InMemorySaver())
    state = {
        "normalized_query": "q",
        "document_catalog": {"doc-1": {"text": "t", "source": "", "score": 1}},
        "cited_id_check_feedback": "Cited document id validation failed.",
        "faithfulness_feedback": "Faithfulness check failed.",
    }
    await graph.answer_node(state)
    assert fake.last_messages is not None
    types = [type(message) for message in fake.last_messages]
    assert SystemMessage in types
    assert HumanMessage in types
    ai_messages = [message for message in fake.last_messages if isinstance(message, AIMessage)]
    assert len(ai_messages) == 2
    assert "Cited document id" in ai_messages[0].content
    assert "Faithfulness" in ai_messages[1].content


@pytest.mark.asyncio
async def test_validate_cited_ids_increments_and_feedback(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "cited_id_retry_max", 10)
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    graph = RetrievalGraph(InMemorySaver())
    state = {
        "document_catalog": {"a": {"text": "t", "source": "", "score": 1}},
        "cited_document_ids": ["missing"],
        "cited_id_check_retry_count": 0,
    }
    result = await graph.validate_cited_ids_node(state)
    assert result["cited_id_check_retry_count"] == 1
    assert "not in document_catalog" in result["cited_id_check_feedback"]
    assert "final_sources" not in result


@pytest.mark.asyncio
async def test_validate_cited_ids_exhausted_sets_empty_sources(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "cited_id_retry_max", 1)
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    graph = RetrievalGraph(InMemorySaver())
    state = {
        "document_catalog": {"a": {"text": "t", "source": "", "score": 1}},
        "cited_document_ids": ["missing"],
        "cited_id_check_retry_count": 0,
    }
    result = await graph.validate_cited_ids_node(state)
    assert result["cited_id_check_retry_count"] == 1
    assert result["final_sources"] == []


@pytest.mark.asyncio
async def test_faithfulness_pass_fills_sources(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    fake = FakeStructuredClient(FaithfulnessResult(passed=True, reason="ok"))
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)

    graph = RetrievalGraph(InMemorySaver())
    answer_msg = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "Because of X",
            "cited_document_ids": ["doc-1"],
            "confidence": "high",
            "sources": [],
        },
    )
    state = {
        "answer_mode": "full",
        "cited_document_ids": ["doc-1"],
        "document_catalog": {
            "doc-1": {"text": "X is true", "source": "hotpotqa", "score": 0.8},
        },
        "messages": [answer_msg],
        "faithfulness_answer_retry_count": 0,
    }
    result = await graph.faithfulness_node(state)
    assert result["faithfulness_ok"] is True
    assert result["final_sources"] == ["hotpotqa"]
    assert result["messages"][0].name == "answer_node"
    assert "hotpotqa" in result["messages"][0].content


@pytest.mark.asyncio
async def test_faithfulness_fail_increments_answer_retry(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    monkeypatch.setattr(settings_module.settings, "answer_retry_max", 5)
    fake = FakeStructuredClient(FaithfulnessResult(passed=False, reason="unsupported claim"))
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)

    graph = RetrievalGraph(InMemorySaver())
    answer_msg = build_node_ai_message(
        node_name="partial_answer_node",
        payload={
            "answer": "guess",
            "cited_document_ids": ["doc-1"],
            "confidence": "low",
            "sources": [],
        },
    )
    state = {
        "answer_mode": "partial",
        "cited_document_ids": ["doc-1"],
        "document_catalog": {
            "doc-1": {"text": "other", "source": "pmc", "score": 0.2},
        },
        "messages": [answer_msg],
        "faithfulness_answer_retry_count": 0,
    }
    result = await graph.faithfulness_node(state)
    assert result["faithfulness_ok"] is False
    assert result["faithfulness_answer_retry_count"] == 1
    assert "unsupported claim" in result["faithfulness_feedback"]
    assert "final_sources" not in result
