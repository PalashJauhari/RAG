"""Map HotpotQA eval JSON contexts to canonical :class:`~benchmarking.hotpotqa.qdrant_payload.ChunkPayload`."""

from __future__ import annotations

import uuid
from typing import Any

from benchmarking.hotpotqa.enrich_text import build_enriched_text
from benchmarking.hotpotqa.qdrant_payload import ChunkPayload


def context_to_chunk(record: dict[str, Any], context: dict[str, Any]) -> tuple[str, ChunkPayload]:
    """Build a stable point id and canonical payload for one HotpotQA context."""

    raw_text = str(context.get("text") or "").strip()
    enrichments = context.get("enrichment") or {}
    if not isinstance(enrichments, dict):
        enrichments = {}

    enriched = build_enriched_text(
        raw_text=raw_text,
        enrichments=enrichments,
        title=str(context.get("title") or ""),
    )
    context_id = str(context["context_id"])
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, context_id))

    payload = ChunkPayload(
        text=enriched,
        enrichments=enrichments,
        additional_metadata={
            "raw_text": raw_text,
            "source": "hotpotqa",
            "context_id": context_id,
            "question_id": record["id"],
            "title": context.get("title"),
            "type": record.get("type"),
            "level": record.get("level"),
            "is_supporting": context.get("is_supporting"),
            "supporting_sentence_ids": context.get("supporting_sentence_ids") or [],
        },
    )
    return point_id, payload
