"""Upload arXiv chunks from ``chunks.json`` to Qdrant."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from ingestion.chunk_media import (
    extract_images_base64,
    extract_table_html,
    with_prefixed_tables,
)
from ingestion.ingestion_config import load_ingestion_config
from ingestion.skip_reference_chunks import skip_reference_chunks
from qdrant_pq import product_quantization_config

jina_request_sem = asyncio.Semaphore(2)


class ChunkPayload(BaseModel):
    """Qdrant point payload (embed and LLM ``text``; ``raw_text`` is unused by the graph)."""

    text: str
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        return self.model_dump()


def pdf_basename(pdf_path: str) -> str:
    """Return basename of a manifest pdf_path string."""

    return pdf_path.replace("\\", "/").split("/")[-1]


def build_manifest_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map PDF filename to arXiv id and abs URL from manifest."""

    lookup: dict[str, dict[str, str]] = {}
    for paper in manifest.get("papers") or []:
        pdf_path = paper.get("pdf_path")
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        abs_url = str(paper.get("abs_url") or "").strip()
        if not pdf_path or not arxiv_id:
            continue
        if not abs_url:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        lookup[pdf_basename(str(pdf_path))] = {
            "arxiv_id": arxiv_id,
            "abs_url": abs_url,
        }
    return lookup


def element_to_chunk(
    file_result: dict[str, Any],
    element: dict[str, Any],
    *,
    manifest_lookup: dict[str, dict[str, str]],
) -> tuple[str, ChunkPayload] | None:
    """Build stable point id and payload for one Unstructured chunk element."""

    raw_text = str(element.get("text") or "").strip()
    images_base64 = extract_images_base64(element)
    table_html = extract_table_html(element)
    if not raw_text and not table_html:
        return None

    embed_text = with_prefixed_tables(raw_text, table_html)
    filename = str(file_result.get("filename") or "")
    element_id = str(element.get("element_id") or "")
    if not element_id:
        return None

    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{element_id}"))

    metadata = element.get("metadata")
    page_number = None
    if isinstance(metadata, dict):
        page_number = metadata.get("page_number")

    paper = manifest_lookup.get(filename) or {}
    payload = ChunkPayload(
        text=embed_text,
        additional_metadata={
            "raw_text": raw_text,
            "source": paper.get("abs_url") or "",
            "filename": filename,
            "element_id": element_id,
            "page_number": page_number,
            "arxiv_id": paper.get("arxiv_id") or "",
            "images_base64": images_base64,
            "table_html": table_html,
        },
    )
    return point_id, payload


async def create_dense_embeddings(
    config,
    texts: list[str],
) -> list[list[float]]:
    """Embed document strings with OpenAI."""

    client = AsyncOpenAI(api_key=config.openai_api_key)
    response = await client.embeddings.create(
        model=config.openai_embedding_model,
        input=texts,
        dimensions=config.openai_embedding_dimensions,
    )
    return [item.embedding for item in response.data]


async def create_late_interaction_embeddings(
    config,
    texts: list[str],
    *,
    input_type: str = "document",
) -> list[list[list[float]]]:
    """Call Jina multi-vector API for ColBERT-style document embeddings."""

    if not config.jina_api_key:
        raise ValueError("JINA_API_KEY is required when USE_LATE_INTERACTION=true")

    payload = {
        "model": config.jina_colbert_model,
        "dimensions": config.jina_colbert_dimensions,
        "input_type": input_type,
        "embedding_type": "float",
        "input": texts,
    }
    headers = {
        "Authorization": f"Bearer {config.jina_api_key}",
        "Content-Type": "application/json",
    }
    max_attempts = 6
    base_delay_seconds = 1.0

    async with jina_request_sem:
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
            for attempt in range(max_attempts):
                response = await client.post(
                    config.jina_multi_vector_url,
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


async def recreate_collection(client: AsyncQdrantClient, config) -> None:
    """Delete existing collection if present, then create a fresh one."""

    name = config.qdrant_collection_name
    if await client.collection_exists(name):
        await client.delete_collection(name)
        print(f"Deleted existing collection {name!r}")

    dense_kwargs: dict[str, Any] = {
        "size": config.openai_embedding_dimensions,
        "distance": models.Distance.COSINE,
    }
    dense_pq = product_quantization_config(config.dense_pq)
    if dense_pq is not None:
        dense_kwargs["quantization_config"] = dense_pq
    vectors_config = {
        config.qdrant_dense_vector_name: models.VectorParams(**dense_kwargs)
    }
    if config.use_late_interaction:
        vectors_config[config.qdrant_colbert_vector_name] = models.VectorParams(
            size=config.jina_colbert_dimensions,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        )

    sparse_vectors_config = None
    if config.use_bm25:
        sparse_vectors_config = {
            config.qdrant_bm25_vector_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }

    await client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"Created collection {name!r} (dense PQ={config.dense_pq})")


async def upsert_chunks(
    client: AsyncQdrantClient,
    config,
    chunks: list[tuple[str, ChunkPayload]],
) -> None:
    """Embed payload text and upsert points in batches."""

    batch_size = config.ingestion_upload_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [payload.text for _, payload in batch]
        dense_vectors = await create_dense_embeddings(config, texts)
        colbert_vectors = None
        if config.use_late_interaction:
            colbert_vectors = await create_late_interaction_embeddings(
                config,
                texts,
                input_type="document",
            )

        points: list[models.PointStruct] = []
        for index, (point_id, payload) in enumerate(batch):
            embed_input = texts[index]
            vector: dict[str, Any] = {config.qdrant_dense_vector_name: dense_vectors[index]}
            if config.use_bm25:
                vector[config.qdrant_bm25_vector_name] = models.Document(
                    text=embed_input,
                    model=config.qdrant_bm25_model,
                )
            if config.use_late_interaction and colbert_vectors:
                vector[config.qdrant_colbert_vector_name] = colbert_vectors[index]

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload.to_qdrant_payload(),
                )
            )

        await client.upsert(
            collection_name=config.qdrant_collection_name,
            points=points,
            wait=True,
        )
        print(f"Uploaded {min(start + batch_size, len(chunks))}/{len(chunks)} points")


def collect_chunks(
    file_results: list[dict[str, Any]],
    manifest_lookup: dict[str, dict[str, str]],
) -> list[tuple[str, ChunkPayload]]:
    """Build all Qdrant points from chunks.json file results."""

    chunks: list[tuple[str, ChunkPayload]] = []
    skipped_references = 0
    for file_result in file_results:
        raw_elements = [
            element
            for element in (file_result.get("elements") or [])
            if isinstance(element, dict)
        ]
        elements, skipped = skip_reference_chunks(raw_elements)
        skipped_references += skipped
        for element in elements:
            row = element_to_chunk(
                file_result,
                element,
                manifest_lookup=manifest_lookup,
            )
            if row is not None:
                chunks.append(row)
    if skipped_references:
        print(f"Skipped {skipped_references} bibliography chunk(s) (not embedded)")
    return chunks


async def run_upload() -> None:
    """Load chunks.json, recreate collection, upsert."""

    config = load_ingestion_config()
    file_results = json.loads(config.chunks_path.read_text(encoding="utf-8"))
    if not isinstance(file_results, list):
        raise ValueError("chunks.json must be a JSON array of file results")

    manifest_lookup: dict[str, dict[str, str]] = {}
    if config.manifest_path.is_file():
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        manifest_lookup = build_manifest_lookup(manifest)

    chunks = collect_chunks(file_results, manifest_lookup)
    if not chunks:
        raise ValueError("No chunks with non-empty text found in chunks.json")

    client = AsyncQdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        cloud_inference=config.use_bm25,
        timeout=config.request_timeout_seconds,
    )
    try:
        await recreate_collection(client, config)
        await upsert_chunks(client, config, chunks)
    finally:
        await client.close()


def main() -> None:
    """CLI entry: upload arXiv chunks to Qdrant."""

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_upload())


if __name__ == "__main__":
    main()
