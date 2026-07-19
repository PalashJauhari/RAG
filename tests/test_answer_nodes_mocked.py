"""Answer / faithfulness nodes with mocked LLM (no network)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
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
        return {"parsed": self.parsed, "raw": AIMessage(content="raw")}


@pytest.mark.asyncio
async def test_answer_node_sets_mode_and_cited_ids(monkeypatch) -> None:
    parsed = FinalAnswer(
        answer="The answer",
        cited_document_ids=["doc-1"],
        confidence="high",
        sources=["should-be-cleared"],
    )
    fake = FakeStructuredClient(parsed)
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)

    graph = RetrievalGraph(InMemorySaver())
    result = await graph.answer_node(
        {
            "normalized_query": "what?",
            "document_catalog": {
                "doc-1": {"text": "evidence", "source": "hotpotqa", "score": 0.9},
            },
            "faithfulness_feedback": "",
        }
    )
    assert result["answer_mode"] == "full"
    assert result["answer_text"] == "The answer"
    assert result["cited_document_ids"] == ["doc-1"]
    assert result["faithfulness_feedback"] == ""
    assert result["messages"][0].name == "answer_node"


@pytest.mark.asyncio
async def test_answer_node_includes_feedback_in_context_when_set(monkeypatch) -> None:
    parsed = FinalAnswer(answer="retry", cited_document_ids=["doc-1"], confidence="low")
    fake = FakeStructuredClient(parsed)
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)

    graph = RetrievalGraph(InMemorySaver())
    await graph.answer_node(
        {
            "normalized_query": "q",
            "document_catalog": {"doc-1": {"text": "t", "source": "", "score": 1}},
            "faithfulness_feedback": "Cited document id validation failed.",
        }
    )
    human = next(message for message in fake.last_messages if isinstance(message, HumanMessage))
    assert "Cited document id validation failed." in human.content
    assert "## Faithfulness feedback" in human.content


@pytest.mark.asyncio
async def test_answer_node_omits_feedback_when_empty(monkeypatch) -> None:
    parsed = FinalAnswer(answer="ok", cited_document_ids=["doc-1"], confidence="high")
    fake = FakeStructuredClient(parsed)
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)

    graph = RetrievalGraph(InMemorySaver())
    await graph.answer_node(
        {
            "normalized_query": "q",
            "document_catalog": {"doc-1": {"text": "t", "source": "", "score": 1}},
            "faithfulness_feedback": "",
        }
    )
    human = next(message for message in fake.last_messages if isinstance(message, HumanMessage))
    assert "## Faithfulness feedback" not in human.content


@pytest.mark.asyncio
async def test_faithfulness_invalid_ids_skips_llm_and_retries(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    monkeypatch.setattr(settings_module.settings, "answer_retry_max", 5)

    def fail_if_llm(**kwargs):
        raise AssertionError("LLM should not run when cited ids are invalid")

    monkeypatch.setattr("graph.graph.get_llm_client", fail_if_llm)

    graph = RetrievalGraph(InMemorySaver())
    answer_msg = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "Because of X",
            "cited_document_ids": ["missing"],
            "confidence": "high",
            "sources": [],
        },
    )
    result = await graph.faithfulness_node(
        {
            "answer_mode": "full",
            "answer_text": "Because of X",
            "cited_document_ids": ["missing"],
            "document_catalog": {"doc-1": {"text": "X", "source": "hotpotqa", "score": 0.8}},
            "messages": [answer_msg],
            "faithfulness_retry_count": 0,
        }
    )
    assert result["faithfulness_ok"] is False
    assert result["faithfulness_retry_count"] == 1
    assert "not in document_catalog" in result["faithfulness_feedback"]
    assert "final_sources" not in result


@pytest.mark.asyncio
async def test_faithfulness_invalid_ids_exhausted_force_pass_valid_sources(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    monkeypatch.setattr(settings_module.settings, "answer_retry_max", 1)

    def fail_if_llm(**kwargs):
        raise AssertionError("LLM should not run when cited ids are invalid")

    monkeypatch.setattr("graph.graph.get_llm_client", fail_if_llm)

    graph = RetrievalGraph(InMemorySaver())
    answer_msg = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "mix",
            "cited_document_ids": ["doc-1", "missing"],
            "confidence": "medium",
            "sources": [],
        },
    )
    result = await graph.faithfulness_node(
        {
            "answer_mode": "full",
            "answer_text": "mix",
            "cited_document_ids": ["doc-1", "missing"],
            "document_catalog": {"doc-1": {"text": "ok", "source": "hotpotqa", "score": 0.9}},
            "messages": [answer_msg],
            "faithfulness_retry_count": 0,
        }
    )
    assert result["faithfulness_ok"] is True
    assert result["faithfulness_retry_count"] == 1
    assert result["final_sources"] == ["hotpotqa"]
    assert result["faithfulness_feedback"] == ""


@pytest.mark.asyncio
async def test_faithfulness_llm_pass_fills_sources(monkeypatch) -> None:
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
    result = await graph.faithfulness_node(
        {
            "answer_mode": "full",
            "answer_text": "Because of X",
            "cited_document_ids": ["doc-1"],
            "document_catalog": {
                "doc-1": {"text": "X is true", "source": "hotpotqa", "score": 0.8},
            },
            "messages": [answer_msg],
            "faithfulness_retry_count": 0,
        }
    )
    assert result["faithfulness_ok"] is True
    assert result["final_sources"] == ["hotpotqa"]
    assert "messages" not in result
    assert "Because of X" in fake.last_messages[1].content


@pytest.mark.asyncio
async def test_faithfulness_llm_fail_increments_retry(monkeypatch) -> None:
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
    result = await graph.faithfulness_node(
        {
            "answer_mode": "partial",
            "answer_text": "guess",
            "cited_document_ids": ["doc-1"],
            "document_catalog": {
                "doc-1": {"text": "other", "source": "pmc", "score": 0.2},
            },
            "messages": [answer_msg],
            "faithfulness_retry_count": 0,
        }
    )
    assert result["faithfulness_ok"] is False
    assert result["faithfulness_retry_count"] == 1
    assert "unsupported claim" in result["faithfulness_feedback"]
    assert "final_sources" not in result


@pytest.mark.asyncio
async def test_faithfulness_llm_fail_exhausted_force_pass(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    monkeypatch.setattr(settings_module.settings, "answer_retry_max", 1)
    fake = FakeStructuredClient(FaithfulnessResult(passed=False, reason="still bad"))
    monkeypatch.setattr("graph.graph.get_llm_client", lambda **kwargs: fake)

    graph = RetrievalGraph(InMemorySaver())
    answer_msg = build_node_ai_message(
        node_name="answer_node",
        payload={
            "answer": "last",
            "cited_document_ids": ["doc-1"],
            "confidence": "low",
            "sources": [],
        },
    )
    result = await graph.faithfulness_node(
        {
            "answer_mode": "full",
            "answer_text": "last",
            "cited_document_ids": ["doc-1"],
            "document_catalog": {
                "doc-1": {"text": "passage", "source": "wiki", "score": 0.5},
            },
            "messages": [answer_msg],
            "faithfulness_retry_count": 0,
        }
    )
    assert result["faithfulness_ok"] is True
    assert result["final_sources"] == ["wiki"]
    assert result["faithfulness_feedback"] == ""
