"""Qdrant collection setup and chunk upsert for HotpotQA benchmark."""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient, models

from benchmarking.hotpotqa.embeddings import (
    create_dense_embeddings,
    create_late_interaction_embeddings,
)
from benchmarking.hotpotqa.qdrant_payload import ChunkPayload, get_enriched_text
from benchmarking.hotpotqa.settings import HotpotQASettings


async def recreate_collection(
    client: AsyncQdrantClient,
    settings: HotpotQASettings,
) -> None:
    """Delete existing collection with the same name, then create a fresh one."""

    name = settings.qdrant_collection_name
    if await client.collection_exists(name):
        await client.delete_collection(name)
        print(f"Deleted existing collection {name!r}")

    vectors_config = {
        settings.qdrant_dense_vector_name: models.VectorParams(
            size=settings.openai_embedding_dimensions,
            distance=models.Distance.COSINE,
        )
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
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"Created collection {name!r}")


async def upsert_chunks(
    client: AsyncQdrantClient,
    settings: HotpotQASettings,
    chunks: list[tuple[str, ChunkPayload]],
    *,
    batch_size: int,
) -> None:
    """Embed ``payload.text`` for each chunk and upsert to Qdrant."""

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [get_enriched_text(payload) for _, payload in batch]
        dense_vectors = await create_dense_embeddings(settings, texts)
        colbert_vectors = None
        if settings.use_late_interaction:
            colbert_vectors = await create_late_interaction_embeddings(
                settings,
                texts,
                input_type="document",
            )

        points: list[models.PointStruct] = []
        for index, (point_id, payload) in enumerate(batch):
            embed_input = texts[index]
            vector: dict[str, Any] = {settings.qdrant_dense_vector_name: dense_vectors[index]}
            if settings.use_bm25:
                vector[settings.qdrant_bm25_vector_name] = models.Document(
                    text=embed_input,
                    model=settings.qdrant_bm25_model,
                )
            if settings.use_late_interaction and colbert_vectors:
                vector[settings.qdrant_colbert_vector_name] = colbert_vectors[index]

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload.to_qdrant_payload(),
                )
            )

        await client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        print(f"Uploaded {min(start + batch_size, len(chunks))}/{len(chunks)} points")
