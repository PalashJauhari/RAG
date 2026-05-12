"""Slim retrieval tool payloads for chat (HotpotQA-style payloads: payload.text)."""

from __future__ import annotations

from typing import Any


def compact_documents_for_llm(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reduce retriever docs to score + passage text only for ToolMessage.content.
    Expects dicts shaped like Retriever output (payload with `text`).
    """

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
        slim.append({"score": doc.get("score"), "text": text})
    return slim
