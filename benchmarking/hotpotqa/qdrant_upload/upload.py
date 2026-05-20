"""Upload HotpotQA paragraph contexts to Qdrant for retrieval eval.

If ``QDRANT_COLLECTION_NAME`` already exists, deletes it and creates a fresh collection.
Point ids are stable ``uuid5`` hashes of ``context_id``. Embeds dense, optional BM25
document vectors, and optional ColBERT document vectors per batch.

Run: ``python -m benchmarking.hotpotqa.qdrant_upload.upload``
"""

import asyncio
import json
import uuid

from qdrant_client import models

from benchmarking.hotpotqa.settings import HotpotQASettings
from retriever.retriever import Retriever


async def recreate_collection(retriever: Retriever) -> None:
    """Delete existing collection (if any), then create with current vector config."""
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


async def main() -> None:
    """Read processed JSON, embed batches, upsert points with benchmark payload metadata."""
    settings = HotpotQASettings()
    retriever = Retriever(settings)
    await recreate_collection(retriever)

    records = json.loads(settings.processed_dataset_path.read_text(encoding="utf-8"))
    docs = []
    for record in records:
        for context in record["contexts"]:
            if context["text"]:
                docs.append(
                    {
                        "point_id": str(
                            uuid.uuid5(uuid.NAMESPACE_URL, context["context_id"])
                        ),
                        "text": context["text"],
                        "payload": {
                            "benchmark": "hotpotqa",
                            "question_id": record["id"],
                            "type": record["type"],
                            "level": record["level"],
                            "context_id": context["context_id"],
                            "title": context["title"],
                            "text": context["text"],
                            "is_supporting": context["is_supporting"],
                            "supporting_sentence_ids": context["supporting_sentence_ids"],
                        },
                    }
                )

    # --- Batched upsert: dense + optional BM25 document + optional ColBERT ---
    batch_size = settings.hotpotqa_upload_batch_size
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        texts = [doc["text"] for doc in batch]
        dense_vectors = await retriever.create_dense_embeddings(texts)
        colbert_vectors = None
        if settings.use_late_interaction:
            colbert_vectors = await retriever.create_late_interaction_embeddings(
                texts,
                input_type="document",
            )

        points = []
        for index, doc in enumerate(batch):
            vector = {settings.qdrant_dense_vector_name: dense_vectors[index]}
            if settings.use_bm25:
                vector[settings.qdrant_bm25_vector_name] = models.Document(
                    text=doc["text"],
                    model=settings.qdrant_bm25_model,
                )
            if settings.use_late_interaction and colbert_vectors:
                vector[settings.qdrant_colbert_vector_name] = colbert_vectors[index]

            points.append(
                models.PointStruct(
                    id=doc["point_id"],
                    vector=vector,
                    payload=doc["payload"],
                )
            )

        await retriever.qdrant.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        print(f"Uploaded {min(start + batch_size, len(docs))}/{len(docs)} points")


if __name__ == "__main__":
    asyncio.run(main())
