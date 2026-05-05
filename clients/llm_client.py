from functools import lru_cache

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from config.settings import settings


@lru_cache(maxsize=1)
def get_openai_client(api_key: str | None = None) -> AsyncOpenAI:
    """Shared OpenAI SDK client for structured outputs and embeddings."""

    return AsyncOpenAI(api_key=api_key or settings.openai_api_key)


@lru_cache(maxsize=2)
def get_chat_llm(json_mode: bool = False) -> ChatOpenAI:
    """Shared LangChain chat model used by the graph orchestrator."""

    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.openai_temperature,
        model_kwargs=model_kwargs,
    )
