from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from config.settings import settings
from middleware.llm_rate_limit import OPENAI_RATE_LIMITER


@lru_cache(maxsize=4)
def get_embeddings_client(api_key: str | None = None) -> AsyncOpenAI:
    """Shared OpenAI SDK client for dense vector embeddings."""

    return AsyncOpenAI(api_key=api_key or settings.openai_api_key)


def get_llm_client(
    *,
    model: str | None = None,
    temperature: float | None = None,
    output_schema: Any | None = None,
    json_mode: bool = False,
) -> Any:
    """Build a ChatOpenAI client with shared rate limiting and optional output schema."""

    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    kwargs: dict[str, Any] = {
        "model": model or settings.openai_llm_model,
        "api_key": settings.openai_api_key,
        "temperature": settings.openai_temperature if temperature is None else temperature,
        "model_kwargs": model_kwargs,
    }
    if OPENAI_RATE_LIMITER is not None:
        kwargs["rate_limiter"] = OPENAI_RATE_LIMITER

    llm = ChatOpenAI(**kwargs)
    if output_schema is not None:
        return llm.with_structured_output(output_schema)
    return llm

