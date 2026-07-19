"""Tests for cited_document_ids validation helpers."""

from __future__ import annotations

from graph.graph import cited_ids_valid


def test_cited_ids_valid_all_present() -> None:
    catalog = {"a": {"text": "t", "source": "", "score": 1}, "b": {"text": "u", "source": "", "score": 1}}
    ok, invalid = cited_ids_valid(["a", "b"], catalog)
    assert ok is True
    assert invalid == []


def test_cited_ids_valid_reports_invalid() -> None:
    catalog = {"a": {"text": "t", "source": "", "score": 1}}
    ok, invalid = cited_ids_valid(["a", "missing", ""], catalog)
    assert ok is False
    assert "missing" in invalid


def test_cited_ids_valid_empty_list_is_ok() -> None:
    ok, invalid = cited_ids_valid([], {"a": {"text": "t", "source": "", "score": 1}})
    assert ok is True
    assert invalid == []
