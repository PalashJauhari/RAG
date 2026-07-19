"""post_deployment_metrics logs to root Langfuse observation only when enabled."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

import pytest

from config import settings as settings_module
from graph.graph import RetrievalGraph, build_node_ai_message
from observability.langfuse_handler import langfuse_root_observation


class FakeRoot:
    def __init__(self) -> None:
        self.output = None

    def update(self, **kwargs):
        self.output = kwargs.get("output")


@pytest.mark.asyncio
async def test_metrics_noop_when_langfuse_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", False)
    graph = RetrievalGraph(InMemorySaver())
    assert await graph.post_deployment_metrics_node({"user_question": "q"}) == {}


@pytest.mark.asyncio
async def test_metrics_updates_root_observation(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "langfuse_tracing_enabled", True)
    root = FakeRoot()
    token = langfuse_root_observation.set(root)
    try:
        graph = RetrievalGraph(InMemorySaver())
        message = build_node_ai_message(
            node_name="answer_node",
            payload={
                "answer": "done",
                "cited_document_ids": ["id-1"],
                "confidence": "high",
                "sources": [],
            },
        )
        result = await graph.post_deployment_metrics_node(
            {
                "user_question": "raw q",
                "normalized_query": "norm q",
                "document_catalog": {
                    "id-1": {"text": "passage", "source": "hotpotqa", "score": 0.9},
                },
                "strategies_used": ["fast_bm25_retrieval", "fast_bm25_late_interaction_retrieval"],
                "retrieval_retry_count": 2,
                "faithfulness_retry_count": 1,
                "facts": [
                    {"fact_id": 1, "fact": "a", "verification_status": True},
                    {"fact_id": 2, "fact": "b", "verification_status": False},
                ],
                "recall_sufficient": False,
                "answer_mode": "partial",
                "answer_text": "done",
                "cited_document_ids": ["id-1"],
                "final_sources": ["hotpotqa"],
                "faithfulness_ok": True,
                "faithfulness_forced_pass": True,
                "graph_failure": {},
                "messages": [message],
            }
        )
        assert result == {}
        assert root.output is not None
        assert root.output["user_question"] == "raw q"
        assert root.output["normalized_query"] == "norm q"
        assert root.output["document_catalog"]["id-1"]["text"] == "passage"
        assert root.output["strategies_used"] == [
            "fast_bm25_retrieval",
            "fast_bm25_late_interaction_retrieval",
        ]
        assert root.output["unsupported_fact_ids"] == [2]
        assert root.output["confidence"] == "high"
        assert root.output["faithfulness_forced_pass"] is True
        assert root.output["answer_mode"] == "partial"
    finally:
        langfuse_root_observation.reset(token)
