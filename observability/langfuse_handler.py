"""Langfuse tracing integration for FastAPI, graph nodes, and LangChain callbacks.

When ``LANGFUSE_TRACING_ENABLED=false``, :func:`get_observe` returns a no-op decorator and
:func:`get_langfuse_callbacks` returns an empty list so call sites need no branches.
"""

import os
from functools import wraps
from typing import Any, Callable

from config.settings import settings


def configure_langfuse_env() -> None:
    """Copy settings into environment variables the Langfuse SDK reads at import time."""
    host = settings.langfuse_host or settings.langfuse_base_url
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", host)
    os.environ.setdefault(
        "LANGFUSE_TRACING_ENABLED",
        "true" if settings.langfuse_tracing_enabled else "false",
    )


def get_langfuse_callbacks() -> list[Any]:
    """Return LangChain ``CallbackHandler`` instances for graph ``ainvoke`` config.

    Returns:
        Empty list when tracing is disabled; otherwise a one-element handler list.
    """
    if not settings.langfuse_tracing_enabled:
        return []
    configure_langfuse_env()
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def get_observe() -> Callable:
    """Return Langfuse ``@observe`` or a transparent no-op decorator.

    Supports both ``@observe`` and ``@observe(name="...")`` usage patterns on graph nodes.
    """
    if not settings.langfuse_tracing_enabled:

        def noop(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def wrapper(fn: Callable) -> Callable:
                return fn

            return wrapper

        return noop

    configure_langfuse_env()
    from langfuse import observe

    return observe
