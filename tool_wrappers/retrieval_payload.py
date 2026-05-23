"""Compact retriever results for graph state and LLM prompts.

Strips Qdrant payload metadata so recall/answer nodes see ``id``, ``score``, and ``text``.
"""

from __future__ import annotations

from typing import Any


def compact_hotqa_documents_for_llm(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce retriever docs to the fields consumed by recall/answer prompts."""

    slim: list[dict[str, Any]] = []
    for doc in documents:
        payload = doc.get("payload") or {}
        raw = payload.get("text")
        if raw is None:
            text = ""
        elif isinstance(raw, str):
            text = raw
        else:
            text = str(raw)
        slim.append({"id": doc.get("id"), "score": doc.get("score"), "text": text})
    return slim
