"""Upload HotpotQA paragraph contexts to Qdrant for retrieval eval.

If ``QDRANT_COLLECTION_NAME`` already exists, deletes it and creates a fresh collection.
Uses the benchmark payload contract in ``benchmarking.hotpotqa.qdrant_payload``.
Embeds ``payload.text``; RAGAS consumes ``additional_metadata.raw_text``.

Run: ``python -m benchmarking.hotpotqa.qdrant_upload.upload``
"""

from __future__ import annotations

import asyncio
import json

from qdrant_client import AsyncQdrantClient

from benchmarking.hotpotqa.adapters.hotpotqa import context_to_chunk
from benchmarking.hotpotqa.qdrant_payload import ChunkPayload
from benchmarking.hotpotqa.qdrant_upload_lib import recreate_collection, upsert_chunks
from benchmarking.hotpotqa.settings import HotpotQASettings


async def main() -> None:
    """Read processed JSON, build canonical chunks, upsert to Qdrant."""

    settings = HotpotQASettings()
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.use_bm25,
        timeout=settings.request_timeout_seconds,
    )
    try:
        await recreate_collection(client, settings)

        records = json.loads(settings.processed_dataset_path.read_text(encoding="utf-8"))
        chunks: list[tuple[str, ChunkPayload]] = []
        for record in records:
            for context in record["contexts"]:
                if context.get("text"):
                    chunks.append(context_to_chunk(record, context))

        await upsert_chunks(
            client,
            settings,
            chunks,
            batch_size=settings.hotpotqa_upload_batch_size,
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
