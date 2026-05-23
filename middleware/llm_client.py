"""Factory helpers for OpenAI chat and embedding clients used across the graph.

Graph nodes call :func:`get_llm_client` with per-node models and Pydantic ``output_schema``
from ``output_validation/``. Embeddings are shared via :func:`get_embeddings_client` for Qdrant dense search.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from config.settings import settings
from middleware.llm_rate_limit import OPENAI_RATE_LIMITER


@lru_cache(maxsize=4)
def get_embeddings_client(api_key: str | None = None) -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client for embedding API calls.

    Args:
        api_key: Override key; defaults to ``settings.openai_api_key``.

    Returns:
        Shared AsyncOpenAI instance (up to 4 distinct keys cached).
    """
    return AsyncOpenAI(api_key=api_key or settings.openai_api_key)


def get_llm_client(
    *,
    model: str,
    temperature: float | None = None,
    output_schema: Any | None = None,
    include_raw: bool = False,
) -> Any:
    """Build ChatOpenAI with optional structured output and global rate limiting.

    When ``output_schema`` is set, returns ``llm.with_structured_output(...)`` so graph
    nodes receive ``{"parsed": Model, "raw": AIMessage}`` when ``include_raw=True``.

    Args:
        model: OpenAI chat model id for this call.
        temperature: Override ``settings.openai_temperature``.
        output_schema: Pydantic model class for structured parsing.
        include_raw: Pass through to ``with_structured_output`` for metadata preservation.

    Returns:
        ChatOpenAI or Runnable with structured output wrapper.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": settings.openai_api_key,
        "temperature": settings.openai_temperature if temperature is None else temperature,
    }
    if OPENAI_RATE_LIMITER is not None:
        kwargs["rate_limiter"] = OPENAI_RATE_LIMITER

    llm = ChatOpenAI(**kwargs)
    # Graph nodes pass Pydantic models from output_validation/; include_raw preserves AIMessage metadata.
    if output_schema is not None:
        return llm.with_structured_output(output_schema, include_raw=include_raw)
    return llm
