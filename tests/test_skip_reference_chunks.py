"""Bibliography skip at arXiv upload (heading + continuation, appendix exit)."""

from __future__ import annotations

from ingestion.skip_reference_chunks import (
    skip_reference_chunks,
    update_reference_skip_state,
)
from ingestion.upload_qdrant_embedding import collect_chunks, select_file_results


def test_body_mentions_references_are_kept() -> None:
    skip, in_refs = update_reference_skip_state(
        "We discuss prior references to attention in Section 2.",
        False,
    )
    assert skip is False
    assert in_refs is False


def test_refs_heading_enters_skip() -> None:
    skip, in_refs = update_reference_skip_state(
        "References\n\n[1] Vaswani et al. Attention is all you need.",
        False,
    )
    assert skip is True
    assert in_refs is True


def test_page_number_then_bibliography() -> None:
    skip, in_refs = update_reference_skip_state(
        "58\n\nBibliography\n\n[1] Turing. Computing machinery.",
        False,
    )
    assert skip is True
    assert in_refs is True


def test_numbered_acknowledgements_then_refs_in_same_chunk_is_skipped() -> None:
    skip, in_refs = update_reference_skip_state(
        "7. Acknowledgements\n\nWe thank collaborators.\n\nReferences\n\nAdam, M. A paper.",
        False,
    )
    assert skip is True
    assert in_refs is True


def test_letter_appendix_heading_exits_skip() -> None:
    skip, in_refs = update_reference_skip_state(
        "A. Details on the Working Agent Definition for this Project\n\nPlans refers to code flow.",
        True,
    )
    assert skip is False
    assert in_refs is False


def test_citation_author_initial_does_not_exit() -> None:
    skip, in_refs = update_reference_skip_state(
        "A. Vaswani, N. Shazeer, Attention is all you need. NeurIPS, 2017.",
        True,
    )
    assert skip is True
    assert in_refs is True
    elements = [
        {"text": "Intro prose about transformers."},
        {"text": "References\n\n[1] Foo."},
        {"text": "[2] Bar, B. Title of paper. arXiv, 2024."},
        {"text": "<table><tr><td>cite</td></tr></table>"},
        {"text": "Appendix A\n\nProof of Lemma 1."},
        {"text": "More appendix math."},
    ]
    kept, skipped = skip_reference_chunks(elements)
    texts = [row["text"] for row in kept]
    assert skipped == 3
    assert texts == [
        "Intro prose about transformers.",
        "Appendix A\n\nProof of Lemma 1.",
        "More appendix math.",
    ]


def test_second_references_section_skipped_again() -> None:
    elements = [
        {"text": "References\n\n[1] Foo."},
        {"text": "Appendix\n\nExtra proofs."},
        {"text": "Bibliography\n\n[1] Foo again."},
    ]
    kept, skipped = skip_reference_chunks(elements)
    assert skipped == 2
    assert [row["text"] for row in kept] == ["Appendix\n\nExtra proofs."]


def test_collect_chunks_skips_bibliography() -> None:
    file_results = [
        {
            "filename": "paper.pdf",
            "elements": [
                {
                    "element_id": "keep",
                    "text": "Method section.",
                    "metadata": {"page_number": 2},
                },
                {
                    "element_id": "drop",
                    "text": "References\n\n[1] X.",
                    "metadata": {"page_number": 9},
                },
            ],
        }
    ]
    lookup = {
        "paper.pdf": {
            "arxiv_id": "1706.03762",
            "abs_url": "https://arxiv.org/abs/1706.03762",
        }
    }
    chunks = collect_chunks(file_results, lookup)
    assert len(chunks) == 1
    assert chunks[0][1].additional_metadata["element_id"] == "keep"


def test_select_file_results_none_keeps_all() -> None:
    rows = [{"filename": "a.pdf"}, {"filename": "b.pdf"}]
    assert select_file_results(rows, None) == rows


def test_select_file_results_max_files_takes_prefix() -> None:
    rows = [{"filename": "a.pdf"}, {"filename": "b.pdf"}, {"filename": "c.pdf"}]
    assert select_file_results(rows, 2) == rows[:2]
