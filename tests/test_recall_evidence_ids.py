"""Recall-check evidence_document_ids schema and catalog membership."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graph.graph import ensure_evidence_document_ids_in_catalog
from output_validation.recall_check import VerifiedFact


def test_verified_fact_supported_requires_ids() -> None:
    with pytest.raises(ValidationError):
        VerifiedFact(
            fact_id=1,
            fact="Whether X",
            verification_status=True,
            evidence_document_ids=[],
        )


def test_verified_fact_unsupported_forbids_ids() -> None:
    with pytest.raises(ValidationError):
        VerifiedFact(
            fact_id=1,
            fact="Whether X",
            verification_status=False,
            evidence_document_ids=["doc-1"],
        )


def test_verified_fact_no_verification_report_field() -> None:
    fact = VerifiedFact(
        fact_id=1,
        fact="Whether X",
        verification_status=True,
        evidence_document_ids=["doc-1"],
    )
    assert fact.evidence_document_ids == ["doc-1"]
    assert "verification_report" not in VerifiedFact.model_fields


def test_ensure_evidence_ids_ok() -> None:
    facts = [
        {
            "fact_id": 1,
            "verification_status": True,
            "evidence_document_ids": ["a", "b"],
        }
    ]
    catalog = {
        "a": {"text": "t", "source": "", "score": 1},
        "b": {"text": "u", "source": "", "score": 1},
    }
    ensure_evidence_document_ids_in_catalog(facts, catalog)


def test_ensure_evidence_ids_raises_on_missing() -> None:
    facts = [
        {
            "fact_id": 1,
            "verification_status": True,
            "evidence_document_ids": ["a", "missing"],
        }
    ]
    catalog = {"a": {"text": "t", "source": "", "score": 1}}
    with pytest.raises(ValueError, match="not in document_catalog"):
        ensure_evidence_document_ids_in_catalog(facts, catalog)


def test_ensure_evidence_ids_empty_when_unsupported_ok() -> None:
    facts = [{"fact_id": 1, "verification_status": False, "evidence_document_ids": []}]
    ensure_evidence_document_ids_in_catalog(facts, {"a": {"text": "t", "source": "", "score": 1}})
