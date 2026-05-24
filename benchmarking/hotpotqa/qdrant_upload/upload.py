"""Upload HotpotQA paragraph contexts to Qdrant for retrieval eval.

If ``QDRANT_COLLECTION_NAME`` already exists, deletes it and creates a fresh collection.
Uses the canonical chunk payload contract (see root README). Embeds ``payload.text``;
graph and RAGAS consume ``additional_metadata.raw_text``.

Run: ``python -m benchmarking.hotpotqa.qdrant_upload.upload``
"""

from __future__ import annotations

import asyncio
import json

from benchmarking.hotpotqa.settings import HotpotQASettings
from ingestion.adapters.hotpotqa import context_to_chunk
from ingestion.qdrant_upload import recreate_collection, upsert_chunks
from ingestion.schema import ChunkPayload
from retriever.retriever import Retriever


async def main() -> None:
    """Read processed JSON, build canonical chunks, upsert to Qdrant."""

    settings = HotpotQASettings()
    retriever = Retriever(settings)
    await recreate_collection(retriever)

    records = json.loads(settings.processed_dataset_path.read_text(encoding="utf-8"))
    chunks: list[tuple[str, ChunkPayload]] = []
    for record in records:
        for context in record["contexts"]:
            if context.get("text"):
                chunks.append(context_to_chunk(record, context))

    await upsert_chunks(
        retriever,
        chunks,
        batch_size=settings.hotpotqa_upload_batch_size,
    )
    await retriever.qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
