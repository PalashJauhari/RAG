"""Tests for document_catalog helpers and retriever hit → catalog conversion."""

from __future__ import annotations

from graph.graph import (
    build_sources_from_catalog,
    catalog_docs_block,
    catalog_texts_for_prompt,
    merge_document_catalog,
)
from tool_wrappers.retrieval_payload import (
    catalog_entries_from_retriever_hits,
    get_raw_text,
    get_source_label,
)


def test_get_raw_text_prefers_additional_metadata() -> None:
    payload = {
        "text": "enriched",
        "additional_metadata": {"raw_text": "raw passage", "source": "hotpotqa"},
    }
    assert get_raw_text(payload) == "raw passage"


def test_get_source_label_empty_when_missing() -> None:
    assert get_source_label({"text": "x"}) == ""
    assert get_source_label({"additional_metadata": {"raw_text": "a"}}) == ""
    assert get_source_label({"additional_metadata": {"source": "pmc"}}) == "pmc"


def test_catalog_entries_from_retriever_hits() -> None:
    hits = [
        {
            "id": "id-1",
            "score": 0.9,
            "payload": {
                "text": "enriched",
                "additional_metadata": {"raw_text": "raw one", "source": "hotpotqa"},
            },
        },
        {
            "id": "id-2",
            "score": 0.5,
            "payload": {
                "text": "only embed text",
                "additional_metadata": {},
            },
        },
    ]
    catalog = catalog_entries_from_retriever_hits(hits)
    assert catalog["id-1"]["text"] == "raw one"
    assert catalog["id-1"]["source"] == "hotpotqa"
    assert catalog["id-1"]["score"] == 0.9
    assert catalog["id-2"]["text"] == "only embed text"
    assert catalog["id-2"]["source"] == ""


def test_merge_document_catalog_keeps_existing_on_collision() -> None:
    existing = {"a": {"text": "old", "source": "s", "score": 1.0}}
    new_rows = {
        "a": {"text": "new", "source": "t", "score": 0.1},
        "b": {"text": "b", "source": "", "score": 0.2},
    }
    merged = merge_document_catalog(existing, new_rows)
    assert merged["a"]["text"] == "old"
    assert merged["b"]["text"] == "b"
    assert list(merged.keys()) == ["a", "b"]


def test_catalog_texts_for_prompt() -> None:
    catalog = {
        "a": {"text": " one ", "source": "x", "score": 1},
        "b": {"text": "", "source": "y", "score": 1},
        "c": {"text": "two", "source": "", "score": 1},
    }
    assert catalog_texts_for_prompt(catalog) == ["one", "two"]


def test_catalog_docs_block_includes_ids() -> None:
    catalog = {"pid": {"text": "hello", "source": "hotpotqa", "score": 0.7}}
    block = catalog_docs_block(catalog)
    assert "pid" in block
    assert "hello" in block
    assert catalog_docs_block({}) == "(none)"


def test_build_sources_from_catalog_skips_empty_and_dedupes() -> None:
    catalog = {
        "a": {"text": "t", "source": "hotpotqa", "score": 1},
        "b": {"text": "t", "source": "", "score": 1},
        "c": {"text": "t", "source": "hotpotqa", "score": 1},
        "d": {"text": "t", "source": "pmc", "score": 1},
    }
    assert build_sources_from_catalog(catalog, ["a", "b", "c", "d", "missing"]) == [
        "hotpotqa",
        "pmc",
    ]
