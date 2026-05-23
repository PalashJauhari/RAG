"""Compact retriever results for graph state and LLM prompts.

Strips Qdrant payload metadata so recall/answer nodes see only ``score`` and ``text``.
Source citation wiring is deferred; extend here when ``sources`` are populated.
"""

from __future__ import annotations

from typing import Any


def compact_hotqa_documents_for_llm(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reduce retriever docs to the fields currently consumed by recall/answer prompts.

    Source metadata is intentionally left out for now; only score and text are exposed.
    """

    slim: list[dict[str, Any]] = []
    for doc in documents:
        # Qdrant returns id/payload/rank; graph state and prompts only need score + text.
        payload = doc.get("payload") or {}
        raw = payload.get("text")
        if raw is None:
            text = ""
        elif isinstance(raw, str):
            text = raw
        else:
            text = str(raw)
        slim.append({"score": doc.get("score"), "text": text})
    return slim
