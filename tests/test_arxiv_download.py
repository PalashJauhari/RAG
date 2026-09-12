"""Tests for arXiv id parsing used by the download pipeline."""

from __future__ import annotations

from ingestion.download_arxiv_pdfs import (
    abs_url_for,
    extract_arxiv_id_from_work,
    normalize_arxiv_id,
    pdf_filename_for,
)


def test_normalize_arxiv_id_from_urls_and_versions() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert normalize_arxiv_id("https://arxiv.org/pdf/1706.03762v7") == "1706.03762"
    assert normalize_arxiv_id("arxiv:2210.03629v3") == "2210.03629"
    assert normalize_arxiv_id("") is None


def test_extract_arxiv_id_from_openalex_work() -> None:
    work = {
        "ids": {"arxiv": "https://arxiv.org/abs/2501.12345"},
        "locations": [],
    }
    assert extract_arxiv_id_from_work(work) == "2501.12345"


def test_abs_url_and_filename() -> None:
    assert abs_url_for("1706.03762") == "https://arxiv.org/abs/1706.03762"
    assert pdf_filename_for("1706.03762") == "arxiv_1706.03762.pdf"
