"""Compact retriever results for graph state and LLM prompts.

Graph ``retrieved_documents`` use ``{id, score, text}`` where ``text`` is always
``additional_metadata.raw_text`` from the Qdrant payload (not the enriched embed string).
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


def compact_documents_for_llm(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce retriever docs to raw passage text for recall/answer prompts."""

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
