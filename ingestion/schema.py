"""Canonical Qdrant chunk payload contract for all corpora.

Qdrant stores ``text`` (embedded string), ``enrichments``, and ``additional_metadata``
with mandatory ``raw_text``. Graph, API, and RAGAS consume ``raw_text`` only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkPayload(BaseModel):
    """Standard payload stored on each Qdrant point."""

    text: str = Field(description="String embedded at upload (enriched when enrichments set).")
    enrichments: dict[str, Any] = Field(default_factory=dict)
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("ChunkPayload.text must be non-empty")
        return value

    @field_validator("additional_metadata")
    @classmethod
    def raw_text_required(cls, value: dict[str, Any]) -> dict[str, Any]:
        raw = value.get("raw_text")
        if raw is None or not str(raw).strip():
            raise ValueError("ChunkPayload.additional_metadata.raw_text is required")
        return value

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize for Qdrant upsert."""

        return self.model_dump()


def get_embed_text(payload: dict[str, Any] | ChunkPayload) -> str:
    """Return the string used for dense/BM25/ColBERT embedding."""

    if isinstance(payload, ChunkPayload):
        return payload.text
    return str((payload or {}).get("text") or "")


def get_raw_text(payload: dict[str, Any] | ChunkPayload) -> str:
    """Return raw passage text for graph, API, and RAGAS."""

    if isinstance(payload, ChunkPayload):
        return str(payload.additional_metadata.get("raw_text") or "")

    data = payload or {}
    additional = data.get("additional_metadata")
    if isinstance(additional, dict):
        raw = additional.get("raw_text")
        if raw is not None and str(raw).strip():
            return str(raw)

    # Legacy HotpotQA flat payload fallback (pre-contract collections).
    return str(data.get("text") or "")


def get_metadata(payload: dict[str, Any] | ChunkPayload) -> dict[str, Any]:
    """Return ``additional_metadata`` dict from a payload."""

    if isinstance(payload, ChunkPayload):
        return dict(payload.additional_metadata)
    additional = (payload or {}).get("additional_metadata")
    return dict(additional) if isinstance(additional, dict) else {}
