"""Recall-check evidence_document_ids schema and catalog membership."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graph.graph import (
    RecallCatalogIdError,
    ensure_evidence_document_ids_in_catalog,
    invalid_evidence_document_ids,
)
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


def test_invalid_evidence_ids_lists_missing_once() -> None:
    facts = [
        {
            "fact_id": 1,
            "evidence_document_ids": ["good-id", "13", "good-id"],
        }
    ]
    assert invalid_evidence_document_ids(facts, {"good-id": {"text": "t"}}) == ["13"]


@pytest.mark.asyncio
async def test_verify_all_facts_retries_bad_ids_with_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    from graph import graph as graph_mod

    calls: list[str] = []

    class _Client:
        async def ainvoke(self, messages: list) -> dict:
            calls.append(str(messages[1].content))
            if len(calls) == 1:
                facts = [
                    VerifiedFact(
                        fact_id=1,
                        fact="Whether X",
                        verification_status=True,
                        evidence_document_ids=["bad-id"],
                    )
                ]
            else:
                facts = [
                    VerifiedFact(
                        fact_id=1,
                        fact="Whether X",
                        verification_status=True,
                        evidence_document_ids=["good-id"],
                    )
                ]
            return {"parsed": type("R", (), {"facts": facts})(), "raw": None}

    monkeypatch.setattr(graph_mod, "get_llm_client", lambda **_kwargs: _Client())
    monkeypatch.setattr(graph_mod.settings, "graph_node_retry_max_attempts", 3)

    out = await graph_mod.verify_all_facts(
        [{"fact_id": 1, "fact": "Whether X"}],
        normalized_query="q",
        catalog={"good-id": {"text": "passage", "score": 1}},
    )
    assert out[0]["evidence_document_ids"] == ["good-id"]
    assert len(calls) == 2
    assert "Id validation feedback" in calls[1]
    assert "bad-id" in calls[1]


@pytest.mark.asyncio
async def test_verify_all_facts_exhaustion_raises_catalog_id_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graph import graph as graph_mod

    class _Client:
        async def ainvoke(self, messages: list) -> dict:
            facts = [
                VerifiedFact(
                    fact_id=1,
                    fact="Whether X",
                    verification_status=True,
                    evidence_document_ids=["bad-id"],
                )
            ]
            return {"parsed": type("R", (), {"facts": facts})(), "raw": None}

    monkeypatch.setattr(graph_mod, "get_llm_client", lambda **_kwargs: _Client())
    monkeypatch.setattr(graph_mod.settings, "graph_node_retry_max_attempts", 2)

    with pytest.raises(RecallCatalogIdError, match="not in document_catalog"):
        await graph_mod.verify_all_facts(
            [{"fact_id": 1, "fact": "Whether X"}],
            normalized_query="q",
            catalog={"good-id": {"text": "passage", "score": 1}},
        )
