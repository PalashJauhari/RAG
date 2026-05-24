"""Build prefixed embedding text from HotpotQA context records."""

from __future__ import annotations

from typing import Any

from ingestion.embed_text import build_embedded_text


def build_embedding_text(context: dict[str, Any]) -> str:
    """Return embedded text for HotpotQA upload (delegates to shared builder)."""

    enrichments = context.get("enrichment")
    if not isinstance(enrichments, dict):
        enrichments = {}
    return build_embedded_text(
        raw_text=str(context.get("text") or ""),
        enrichments=enrichments,
        title=str(context.get("title") or ""),
    )
