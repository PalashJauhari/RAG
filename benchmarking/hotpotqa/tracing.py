"""Optional Langfuse tracing for HotpotQA prepare-time LLM calls."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from observability.langfuse_handler import (
    flush_langfuse,
    get_langfuse_client,
    update_llm_generation,
)


def langfuse_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "langfuse_tracing_enabled", False))


def get_client(settings: Any):
    if not langfuse_enabled(settings):
        return None
    return get_langfuse_client()


def record_generation(gen_span: Any, *, model: str, raw: AIMessage | None) -> None:
    update_llm_generation(gen_span, model=model, raw=raw)


def start_parallel_generation(
    parent_span: Any,
    *,
    name: str,
    model: str,
    input_text: str,
    metadata: dict[str, Any],
) -> Any:
    """Create a generation child on ``parent_span`` (no OTel context required)."""
    return parent_span.start_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input_text,
        metadata=metadata,
    )


def finish_generation(
    gen: Any,
    *,
    model: str,
    raw: AIMessage | None,
    output: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record token usage, output, and end a manually created generation span."""
    record_generation(gen, model=model, raw=raw)
    gen.update(output=output, metadata=metadata or {})
    gen.end()


def flush() -> None:
    flush_langfuse()
