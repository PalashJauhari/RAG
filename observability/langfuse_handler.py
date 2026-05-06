import os
from typing import Any

from config.settings import settings


def configure_langfuse_env() -> None:
    """Expose Langfuse settings using names the SDK reads."""

    host = settings.langfuse_host or settings.langfuse_base_url
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", host)
    os.environ.setdefault(
        "LANGFUSE_TRACING_ENABLED",
        "true" if settings.langfuse_tracing_enabled else "false",
    )


def get_langfuse_callbacks() -> list[Any]:
    """Return LangChain callbacks when Langfuse tracing is enabled."""

    if not settings.langfuse_tracing_enabled:
        return []
    configure_langfuse_env()
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]

