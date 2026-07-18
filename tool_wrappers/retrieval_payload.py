"""Build document_catalog entries from retriever hits for graph state and prompts.

Each catalog value is ``{text, source, score}`` where ``text`` is always
``additional_metadata.raw_text`` from the Qdrant payload (not the enriched embed string)
and ``source`` is ``additional_metadata.source`` or ``""``.
"""

from __future__ import annotations

from typing import Any


def get_raw_text(payload: dict[str, Any]) -> str:
    """Return raw passage text from a Qdrant payload dict."""

    data = payload or {}
    additional = data.get("additional_metadata")
    if isinstance(additional, dict):
        raw = additional.get("raw_text")
        if raw is not None and str(raw).strip():
            return str(raw)

    return str(data.get("text") or "")


def get_source_label(payload: dict[str, Any]) -> str:
    """Return ``additional_metadata.source`` or empty string (no fallbacks)."""

    data = payload or {}
    additional = data.get("additional_metadata")
    if isinstance(additional, dict):
        source = additional.get("source")
        if source is not None:
            return str(source).strip()
    return ""


def catalog_entries_from_retriever_hits(
    documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map retriever hits to document_catalog rows keyed by point id."""

    catalog: dict[str, dict[str, Any]] = {}
    for doc in documents:
        point_id = str(doc.get("id") or "").strip()
        if not point_id:
            continue
        payload = doc.get("payload") or {}
        catalog[point_id] = {
            "text": get_raw_text(payload),
            "source": get_source_label(payload),
            "score": doc.get("score"),
        }
    return catalog


def compact_documents_for_llm(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deprecated list shape; prefer :func:`catalog_entries_from_retriever_hits`."""

    slim: list[dict[str, Any]] = []
    for doc in documents:
        payload = doc.get("payload") or {}
        slim.append(
            {
                "id": doc.get("id"),
                "score": doc.get("score"),
                "text": get_raw_text(payload),
            }
        )
    return slim


def compact_hotqa_documents_for_llm(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deprecated alias for :func:`compact_documents_for_llm`."""

    return compact_documents_for_llm(documents)
