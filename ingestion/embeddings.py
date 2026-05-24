"""OpenAI dense and Jina ColBERT embeddings for ingestion upload."""

from __future__ import annotations

import asyncio

import httpx
from openai import AsyncOpenAI

from ingestion.settings import IngestionSettings

_jina_request_sem = asyncio.Semaphore(2)


async def create_dense_embeddings(
    settings: IngestionSettings,
    texts: list[str],
) -> list[list[float]]:
    """Embed document strings with the configured OpenAI embedding model."""

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
        dimensions=settings.openai_embedding_dimensions,
    )
    return [item.embedding for item in response.data]


async def create_late_interaction_embeddings(
    settings: IngestionSettings,
    texts: list[str],
    *,
    input_type: str = "document",
) -> list[list[list[float]]]:
    """Call Jina multi-vector API for ColBERT-style document embeddings."""

    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY is required when USE_LATE_INTERACTION=true")

    payload = {
        "model": settings.jina_colbert_model,
        "dimensions": settings.jina_colbert_dimensions,
        "input_type": input_type,
        "embedding_type": "float",
        "input": texts,
    }
    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
    }
    max_attempts = 6
    base_delay_seconds = 1.0

    async with _jina_request_sem:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            for attempt in range(max_attempts):
                response = await client.post(
                    settings.jina_multi_vector_url,
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 429 and attempt < max_attempts - 1:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = min(60.0, base_delay_seconds * (2**attempt))
                    else:
                        wait = min(60.0, base_delay_seconds * (2**attempt))
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                return [row["embeddings"] for row in response.json()["data"]]
