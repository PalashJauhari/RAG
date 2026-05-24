"""Shared Qdrant collection setup and chunk upsert for all corpora."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from ingestion.schema import ChunkPayload, get_embed_text
from retriever.retriever import Retriever


async def recreate_collection(retriever: Retriever) -> None:
    """Delete existing collection with the same name, then create a fresh one."""

    settings = retriever.config
    name = settings.qdrant_collection_name
    if await retriever.qdrant.collection_exists(name):
        await retriever.qdrant.delete_collection(name)
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

    await retriever.qdrant.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"Created collection {name!r}")


async def upsert_chunks(
    retriever: Retriever,
    chunks: list[tuple[str, ChunkPayload]],
    *,
    batch_size: int = 64,
) -> None:
    """Embed ``payload.text`` for each chunk and upsert to Qdrant."""

    settings = retriever.config
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [get_embed_text(payload) for _, payload in batch]
        dense_vectors = await retriever.create_dense_embeddings(texts)
        colbert_vectors = None
        if settings.use_late_interaction:
            colbert_vectors = await retriever.create_late_interaction_embeddings(
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

        await retriever.qdrant.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        print(f"Uploaded {min(start + batch_size, len(chunks))}/{len(chunks)} points")
