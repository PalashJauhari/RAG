"""Langfuse tracing for the RAG graph.

One root span per ``run`` / ``stream_run`` / ``resume``; inline node spans and nested
``{node}-llm`` generation spans (token counts + model only on generations). Gated by
``LANGFUSE_TRACING_ENABLED`` in :mod:`config.settings`. No ``CallbackHandler`` or ``@observe``.

**Retrieval span** (``retrieval_node`` in :mod:`graph.graph`): ``output`` includes
``strategy``, ``queries``, ``new_doc_count``, ``rows_to_add`` (unique docs appended this pass),
and ``retrieved_documents`` (full accumulated corpus after merge). Passage text can be large.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage
from langfuse import get_client

from config.settings import settings


def is_tracing_enabled() -> bool:
    """Return whether Langfuse tracing is enabled via settings."""
    return settings.langfuse_tracing_enabled


def configure_langfuse_env() -> None:
    """Copy settings into environment variables the Langfuse SDK reads."""
    base_url = settings.langfuse_base_url or settings.langfuse_host
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if base_url:
        os.environ.setdefault("LANGFUSE_BASE_URL", base_url)
        os.environ.setdefault("LANGFUSE_HOST", base_url)
    os.environ.setdefault(
        "LANGFUSE_TRACING_ENABLED",
        "true" if settings.langfuse_tracing_enabled else "false",
    )


def get_langfuse_client() -> Any | None:
    """Return the Langfuse client when tracing is enabled; otherwise ``None``.

    Does not call ``get_client()`` when ``LANGFUSE_TRACING_ENABLED`` is false.
    """
    if not settings.langfuse_tracing_enabled:
        return None
    configure_langfuse_env()
    return get_client()


def flush_langfuse() -> None:
    """Flush buffered observations after a traced invoke; no-op when tracing is disabled."""
    if not settings.langfuse_tracing_enabled:
        return
    get_client().flush()


def llm_token_counts(raw: AIMessage | None) -> tuple[int | None, int | None]:
    """Return ``(input_tokens, output_tokens)`` from LangChain ``usage_metadata``.

    Used for generation-span ``update(input=..., output=...)`` (token counts only).
    """
    if raw is None:
        return None, None
    usage = getattr(raw, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    in_val = int(input_tokens) if input_tokens is not None else None
    out_val = int(output_tokens) if output_tokens is not None else None
    return in_val, out_val
