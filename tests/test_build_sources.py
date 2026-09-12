"""Tests for source list construction from document_catalog."""

from __future__ import annotations

from graph.graph import build_sources_from_catalog


def test_build_sources_preserves_first_seen_order() -> None:
    catalog = {
        "1": {"text": "a", "source": "alpha", "score": 1},
        "2": {"text": "b", "source": "beta", "score": 1},
        "3": {"text": "c", "source": "gamma", "score": 1},
    }
    assert build_sources_from_catalog(catalog, ["2", "1", "3"]) == ["2", "1", "3"]


def test_build_sources_empty_catalog() -> None:
    assert build_sources_from_catalog({}, ["x"]) == []
    assert build_sources_from_catalog(None, None) == []
