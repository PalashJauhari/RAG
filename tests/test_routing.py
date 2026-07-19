"""Routing after recall and faithfulness."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from config import settings as settings_module
from graph.graph import RetrievalGraph


def test_route_after_faithfulness_pass_ends() -> None:
    graph = RetrievalGraph(InMemorySaver())
    assert graph.route_after_faithfulness({"faithfulness_ok": True}) == "end"


def test_route_after_faithfulness_fail_retries_full_mode() -> None:
    graph = RetrievalGraph(InMemorySaver())
    assert (
        graph.route_after_faithfulness(
            {"faithfulness_ok": False, "answer_mode": "full", "faithfulness_retry_count": 1}
        )
        == "answer"
    )


def test_route_after_faithfulness_fail_retries_partial_mode() -> None:
    graph = RetrievalGraph(InMemorySaver())
    assert (
        graph.route_after_faithfulness(
            {"faithfulness_ok": False, "answer_mode": "partial", "faithfulness_retry_count": 2}
        )
        == "partial_answer"
    )


def test_route_after_recall_sufficient_goes_to_answer(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "retrieval_loop_max_retries", 3)
    graph = RetrievalGraph(InMemorySaver())
    assert graph.route_after_recall_check({"recall_sufficient": True, "retrieval_retry_count": 0}) == "answer"


def test_route_after_recall_budget_goes_to_partial(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "retrieval_loop_max_retries", 3)
    graph = RetrievalGraph(InMemorySaver())
    assert (
        graph.route_after_recall_check({"recall_sufficient": False, "retrieval_retry_count": 3})
        == "partial_answer"
    )
