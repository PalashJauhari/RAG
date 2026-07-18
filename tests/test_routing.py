"""Router unit tests for validate / faithfulness / answer_mode."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from config import settings as settings_module
from graph.graph import RetrievalGraph


def make_graph() -> RetrievalGraph:
    return RetrievalGraph(InMemorySaver())


def test_route_after_validate_valid_goes_to_faithfulness(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "cited_id_retry_max", 10)
    graph = make_graph()
    state = {
        "document_catalog": {"a": {"text": "t", "source": "", "score": 1}},
        "cited_document_ids": ["a"],
        "cited_id_check_retry_count": 0,
        "answer_mode": "full",
    }
    assert graph.route_after_validate_cited_ids(state) == "faithfulness"


def test_route_after_validate_invalid_retries_same_mode(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "cited_id_retry_max", 10)
    graph = make_graph()
    state = {
        "document_catalog": {"a": {"text": "t", "source": "", "score": 1}},
        "cited_document_ids": ["missing"],
        "cited_id_check_retry_count": 1,
        "answer_mode": "partial",
    }
    assert graph.route_after_validate_cited_ids(state) == "partial_answer"


def test_route_after_validate_exhausted_ends(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "cited_id_retry_max", 2)
    graph = make_graph()
    state = {
        "document_catalog": {"a": {"text": "t", "source": "", "score": 1}},
        "cited_document_ids": ["missing"],
        "cited_id_check_retry_count": 2,
        "answer_mode": "full",
    }
    assert graph.route_after_validate_cited_ids(state) == "end"


def test_route_after_faithfulness_pass_ends(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "answer_retry_max", 5)
    graph = make_graph()
    assert graph.route_after_faithfulness({"faithfulness_ok": True, "faithfulness_answer_retry_count": 0}) == "end"


def test_route_after_faithfulness_fail_retries_mode(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "answer_retry_max", 5)
    graph = make_graph()
    state = {
        "faithfulness_ok": False,
        "faithfulness_answer_retry_count": 1,
        "answer_mode": "full",
    }
    assert graph.route_after_faithfulness(state) == "answer"


def test_route_answer_mode_helper() -> None:
    graph = make_graph()
    assert graph.route_answer_mode({"answer_mode": "partial"}) == "partial_answer"
    assert graph.route_answer_mode({"answer_mode": "full"}) == "answer"
