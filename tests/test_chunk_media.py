"""Chunk image/table extraction and prefixed embed text."""

from __future__ import annotations

from ingestion.chunk_media import (
    TABLE_HTML_PREFIX,
    extract_images_base64,
    extract_table_html,
    with_prefixed_tables,
)
from ingestion.upload_qdrant_embedding import element_to_chunk


def _chunk_element() -> dict:
    return {
        "element_id": "el-1",
        "text": "Caption of a figure. Some prose.",
        "metadata": {
            "page_number": 4,
            "images": [
                {
                    "element_id": "img-1",
                    "mime_type": "image/png",
                    "base64": "aaa",
                    "page_number": 4,
                }
            ],
            "orig_elements": [
                {
                    "type": "Image",
                    "element_id": "img-1",
                    "metadata": {"page_number": 4},
                },
                {
                    "type": "Table",
                    "element_id": "tbl-1",
                    "metadata": {
                        "page_number": 4,
                        "text_as_html": "<table><tr><td>1</td></tr></table>",
                    },
                },
            ],
        },
    }


def test_extract_images_and_tables() -> None:
    element = _chunk_element()
    assert extract_images_base64(element) == ["aaa"]
    assert extract_table_html(element) == ["<table><tr><td>1</td></tr></table>"]


def test_extract_missing_media_is_none() -> None:
    element = {"text": "only text", "metadata": {"orig_elements": []}}
    assert extract_images_base64(element) is None
    assert extract_table_html(element) is None
    assert extract_images_base64({}) is None
    assert extract_table_html({"metadata": "bad"}) is None


def test_with_prefixed_tables() -> None:
    tables = ["<table>a</table>", "<table>b</table>"]
    out = with_prefixed_tables("prose", tables)
    assert out.startswith("prose")
    assert TABLE_HTML_PREFIX in out
    assert "<table>a</table>" in out
    assert "<table>b</table>" in out
    assert with_prefixed_tables("prose", None) == "prose"
    assert with_prefixed_tables("", ["<table>x</table>"]).startswith(TABLE_HTML_PREFIX)


def test_element_to_chunk_payload_text_and_unused_raw_text() -> None:
    lookup = {
        "paper.pdf": {
            "arxiv_id": "1706.03762",
            "abs_url": "https://arxiv.org/abs/1706.03762",
        }
    }
    result = element_to_chunk(
        {"filename": "paper.pdf"},
        _chunk_element(),
        manifest_lookup=lookup,
    )
    assert result is not None
    _, payload = result
    dumped = payload.to_qdrant_payload()
    meta = dumped["additional_metadata"]
    assert meta["raw_text"] == "Caption of a figure. Some prose."
    assert TABLE_HTML_PREFIX not in meta["raw_text"]
    assert dumped["text"].startswith("Caption of a figure. Some prose.")
    assert TABLE_HTML_PREFIX in dumped["text"]
    assert "<table><tr><td>1</td></tr></table>" in dumped["text"]
    assert meta["images_base64"] == ["aaa"]
    assert meta["table_html"] == ["<table><tr><td>1</td></tr></table>"]
    assert meta["source"] == "https://arxiv.org/abs/1706.03762"
    assert meta["page_number"] == 4


def test_element_to_chunk_null_media_when_absent() -> None:
    result = element_to_chunk(
        {"filename": "x.pdf"},
        {"element_id": "el", "text": "hello", "metadata": {}},
        manifest_lookup={},
    )
    assert result is not None
    _, payload = result
    meta = payload.to_qdrant_payload()["additional_metadata"]
    assert meta["images_base64"] is None
    assert meta["table_html"] is None
    assert payload.text == "hello"
    assert meta["raw_text"] == "hello"


def test_element_to_chunk_skips_empty() -> None:
    assert (
        element_to_chunk(
            {"filename": "x.pdf"},
            {"element_id": "el", "text": "  ", "metadata": {}},
            manifest_lookup={},
        )
        is None
    )
