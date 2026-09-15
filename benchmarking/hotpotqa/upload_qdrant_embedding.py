"""Upload HotpotQA contexts to Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from benchmarking.hotpotqa.benchmark_config import (
    PROCESSED_DATASET_PATH,
    collection_name_for_pq,
    parse_pq_list,
)
from config.settings import settings
from qdrant_pq import product_quantization_config

UPLOAD_BATCH_SIZE = 16

jina_request_sem = asyncio.Semaphore(2)


class ChunkPayload(BaseModel):
    """Qdrant point payload for one HotpotQA context."""

    text: str
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        return self.model_dump()


def context_to_chunk(record: dict[str, Any], context: dict[str, Any]) -> tuple[str, ChunkPayload]:
    """Build stable point id and payload for one context."""

    raw_text = str(context.get("text") or "").strip()
    context_id = str(context["context_id"])
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, context_id))

    payload = ChunkPayload(
        text=raw_text,
        additional_metadata={
            "raw_text": raw_text,
            "source": "hotpotqa",
            "context_id": context_id,
            "question_id": record["id"],
            "title": context.get("title"),
            "type": record.get("type"),
            "level": record.get("level"),
            "is_supporting": context.get("is_supporting"),
            "supporting_sentence_ids": context.get("supporting_sentence_ids") or [],
            "images_base64": None,
            "table_html": None,
        },
    )
    return point_id, payload


async def create_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed document strings with OpenAI."""

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
        dimensions=settings.openai_embedding_dimensions,
    )
    return [item.embedding for item in response.data]


async def create_late_interaction_embeddings(
    texts: list[str],
    *,
    input_type: str = "document",
) -> list[list[list[float]]]:
    """Call Jina multi-vector API for ColBERT-style embeddings."""

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

    async with jina_request_sem:
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
    raise RuntimeError("Jina embedding request failed")


async def recreate_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    dense_pq: str,
) -> None:
    """Delete existing collection if present, then create a fresh one."""

    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)
        print(f"Deleted existing collection {collection_name!r}")

    dense_kwargs: dict[str, Any] = {
        "size": settings.openai_embedding_dimensions,
        "distance": models.Distance.COSINE,
    }
    dense_pq_config = product_quantization_config(dense_pq)
    if dense_pq_config is not None:
        dense_kwargs["quantization_config"] = dense_pq_config
    vectors_config = {
        settings.qdrant_dense_vector_name: models.VectorParams(**dense_kwargs)
    }
    if settings.use_late_interaction:
        vectors_config[settings.qdrant_colbert_vector_name] = models.VectorParams(
            size=settings.jina_colbert_dimensions,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        )

    sparse_vectors_config = None
    if settings.use_bm25:
        sparse_vectors_config = {
            settings.qdrant_bm25_vector_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"Created collection {collection_name!r} (dense PQ={dense_pq})")


async def embed_chunks(
    chunks: list[tuple[str, ChunkPayload]],
) -> list[tuple[str, ChunkPayload, list[float], list[list[float]] | None]]:
    """Embed all chunks once (dense + optional ColBERT) for reuse across PQ collections."""

    embedded: list[tuple[str, ChunkPayload, list[float], list[list[float]] | None]] = []
    for start in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        batch = chunks[start : start + UPLOAD_BATCH_SIZE]
        texts = [payload.text for _, payload in batch]
        dense_vectors = await create_dense_embeddings(texts)
        colbert_vectors = None
        if settings.use_late_interaction:
            colbert_vectors = await create_late_interaction_embeddings(
                texts,
                input_type="document",
            )
        for index, (point_id, payload) in enumerate(batch):
            colbert = colbert_vectors[index] if colbert_vectors else None
            embedded.append((point_id, payload, dense_vectors[index], colbert))
        print(f"Embedded {min(start + UPLOAD_BATCH_SIZE, len(chunks))}/{len(chunks)} points")
    return embedded


async def upsert_embedded_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    embedded: list[tuple[str, ChunkPayload, list[float], list[list[float]] | None]],
) -> None:
    """Upsert precomputed vectors into one collection."""

    for start in range(0, len(embedded), UPLOAD_BATCH_SIZE):
        batch = embedded[start : start + UPLOAD_BATCH_SIZE]
        points: list[models.PointStruct] = []
        for point_id, payload, dense_vector, colbert_vector in batch:
            vector: dict[str, Any] = {settings.qdrant_dense_vector_name: dense_vector}
            if settings.use_bm25:
                vector[settings.qdrant_bm25_vector_name] = models.Document(
                    text=payload.text,
                    model=settings.qdrant_bm25_model,
                )
            if settings.use_late_interaction and colbert_vector is not None:
                vector[settings.qdrant_colbert_vector_name] = colbert_vector
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload.to_qdrant_payload(),
                )
            )
        await client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        print(
            f"Uploaded {min(start + UPLOAD_BATCH_SIZE, len(embedded))}/{len(embedded)} "
            f"points to {collection_name!r}"
        )


async def run_upload(
    *,
    collection_base: str,
    dense_pq_values: list[str],
) -> None:
    """Load JSON, embed once, recreate each PQ collection, upsert."""

    records = json.loads(PROCESSED_DATASET_PATH.read_text(encoding="utf-8"))

    chunks: list[tuple[str, ChunkPayload]] = []
    for record in records:
        for context in record.get("contexts") or []:
            if context.get("text"):
                chunks.append(context_to_chunk(record, context))

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.use_bm25,
        timeout=settings.request_timeout_seconds,
    )
    try:
        embedded = await embed_chunks(chunks)
        for dense_pq in dense_pq_values:
            name = collection_name_for_pq(collection_base, dense_pq)
            await recreate_collection(client, name, dense_pq)
            await upsert_embedded_chunks(client, name, embedded)
    finally:
        await client.close()


def main() -> None:
    """CLI entry: upload HotpotQA contexts to one or more PQ collections."""

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Upload HotpotQA contexts to Qdrant")
    parser.add_argument(
        "--collection-base",
        required=True,
        help="Collection prefix; each PQ becomes {base}_{none|pq8|pq16|pq32}",
    )
    parser.add_argument(
        "--pq",
        nargs="+",
        default=["none"],
        help="PQ modes to create in one go (none pq8 pq16 pq32). Default: none",
    )
    args = parser.parse_args()
    asyncio.run(
        run_upload(
            collection_base=args.collection_base,
            dense_pq_values=parse_pq_list(args.pq),
        )
    )


if __name__ == "__main__":
    main()
