"""Tests for document_catalog helpers and retriever hit → catalog conversion."""

from __future__ import annotations

from graph.graph import (
    build_sources_from_catalog,
    catalog_docs_block,
    catalog_for_llm,
    catalog_texts_for_prompt,
    merge_document_catalog,
)
from tool_wrappers.retrieval_payload import (
    catalog_entries_from_retriever_hits,
    catalog_for_client,
    catalog_without_images,
    cited_ui_entries,
    get_payload_text,
)


def test_get_payload_text_prefers_payload_text() -> None:
    payload = {
        "text": "embed + tables",
        "additional_metadata": {"raw_text": "raw passage"},
    }
    assert get_payload_text(payload) == "embed + tables"


def test_get_payload_text_falls_back_to_raw_text() -> None:
    payload = {
        "text": "  ",
        "additional_metadata": {"raw_text": "legacy raw"},
    }
    assert get_payload_text(payload) == "legacy raw"


def test_catalog_entries_from_retriever_hits() -> None:
    hits = [
        {
            "id": "id-1",
            "score": 0.9,
            "payload": {
                "text": "embed one",
                "additional_metadata": {
                    "raw_text": "raw one",
                    "source": "https://arxiv.org/abs/1706.03762",
                    "page_number": 3,
                    "images_base64": ["img"],
                    "table_html": ["<table></table>"],
                },
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
        {
            "id": "id-3",
            "score": 0.1,
            "payload": {
                "text": "hotpot",
                "additional_metadata": {
                    "source": "hotpotqa",
                    "images_base64": None,
                    "table_html": None,
                },
            },
        },
    ]
    catalog = catalog_entries_from_retriever_hits(hits)
    assert catalog["id-1"]["text"] == "embed one"
    assert catalog["id-1"]["score"] == 0.9
    assert catalog["id-1"]["source"] == "https://arxiv.org/abs/1706.03762"
    assert catalog["id-1"]["page_number"] == 3
    assert catalog["id-1"]["images_base64"] == ["img"]
    assert catalog["id-1"]["table_html"] == ["<table></table>"]
    assert catalog["id-2"] == {
        "text": "only embed text",
        "score": 0.5,
        "source": "",
        "page_number": None,
        "images_base64": None,
        "table_html": None,
    }
    assert catalog["id-3"]["images_base64"] is None
    assert catalog["id-3"]["table_html"] is None
    assert catalog["id-3"]["source"] == "hotpotqa"


def test_catalog_entries_missing_payload_does_not_raise() -> None:
    catalog = catalog_entries_from_retriever_hits(
        [{"id": "x", "score": 1, "payload": None}, {"id": "", "score": 1}]
    )
    assert catalog["x"]["text"] == ""
    assert catalog["x"]["images_base64"] is None
    assert "" not in catalog


def test_merge_document_catalog_keeps_existing_on_collision() -> None:
    existing = {"a": {"text": "old", "score": 1.0}}
    new_rows = {
        "a": {"text": "new", "score": 0.1},
        "b": {"text": "b", "score": 0.2},
    }
    merged = merge_document_catalog(existing, new_rows)
    assert merged["a"]["text"] == "old"
    assert merged["b"]["text"] == "b"
    assert list(merged.keys()) == ["a", "b"]


def test_catalog_texts_for_prompt() -> None:
    catalog = {
        "a": {"text": " one ", "score": 1},
        "b": {"text": "", "score": 1},
        "c": {"text": "two", "score": 1},
    }
    assert catalog_texts_for_prompt(catalog) == ["one", "two"]


def test_catalog_docs_block_omits_source_url() -> None:
    catalog = {
        "pid": {
            "text": "hello",
            "source": "https://arxiv.org/abs/1706.03762",
            "score": 0.7,
            "page_number": 2,
        }
    }
    block = catalog_docs_block(catalog)
    assert "pid" in block
    assert "hello" in block
    assert "arxiv.org" not in block
    assert catalog_docs_block({}) == "(none)"


def test_catalog_for_llm_is_text_and_score_by_id() -> None:
    catalog = {
        "pid": {
            "text": "hello",
            "source": "https://arxiv.org/abs/1706.03762",
            "score": 0.7,
            "page_number": 2,
            "images_base64": ["secret"],
            "table_html": ["<table></table>"],
        }
    }
    slim = catalog_for_llm(catalog)
    assert slim == {"pid": {"text": "hello", "score": 0.7}}
    assert "images_base64" not in slim["pid"]
    assert "table_html" not in slim["pid"]
    assert "source" not in slim["pid"]


def test_catalog_without_images_strips_base64_keeps_tables() -> None:
    catalog = {
        "pid": {
            "text": "t",
            "score": 1,
            "images_base64": ["abc"],
            "table_html": ["<table></table>"],
        }
    }
    stripped = catalog_without_images(catalog)
    assert "images_base64" not in stripped["pid"]
    assert stripped["pid"]["table_html"] == ["<table></table>"]
    assert catalog["pid"]["images_base64"] == ["abc"]


def test_cited_ui_and_client_catalog_skip_missing() -> None:
    catalog = {
        "a": {
            "text": "should not appear",
            "source": "https://arxiv.org/abs/1",
            "page_number": 2,
            "images_base64": ["img"],
            "table_html": ["<table>t</table>"],
        },
        "b": {"text": "hotpot", "source": "hotpotqa", "images_base64": None, "table_html": None},
        "c": {"text": "uncited", "source": "https://example.com", "images_base64": ["nope"]},
        "d": {"text": "empty"},
    }
    ui = cited_ui_entries(catalog, ["a", "b", "missing", "a", "d"])
    assert ui[0]["source"] == "https://arxiv.org/abs/1"
    assert ui[0]["page_number"] == 2
    assert ui[0]["images_base64"] == ["img"]
    assert ui[0]["table_html"] == ["<table>t</table>"]
    assert "text" not in ui[0]
    assert ui[1] == {"source": "hotpotqa"}
    assert len(ui) == 2

    client = catalog_for_client(catalog, ["a", "c"])
    assert "text" not in client["a"]
    assert client["c"]["images_base64"] == ["nope"]
    assert "b" not in client
    assert catalog_for_client({}, ["a"]) == {}
    assert cited_ui_entries(None, None) == []


def test_build_sources_from_catalog_returns_cited_ids() -> None:
    catalog = {
        "a": {"text": "t", "score": 1},
        "b": {"text": "t", "score": 1},
        "c": {"text": "t", "score": 1},
        "d": {"text": "t", "score": 1},
    }
    assert build_sources_from_catalog(catalog, ["a", "b", "c", "a", "missing"]) == [
        "a",
        "b",
        "c",
    ]
