"""Upload arXiv chunks from ``chunks.json`` to Qdrant (optional LLM enrichment)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient, models

from ingestion.chunk_media import (
    extract_images_base64,
    extract_table_html,
    with_prefixed_tables,
)
from ingestion.ingestion_config import load_ingestion_config
from qdrant_pq import product_quantization_config

logger = logging.getLogger(__name__)

ENRICH_CONCURRENCY = 10
ENRICH_MAX_RETRIES = 3

ENRICHMENT_SYSTEM_PROMPT = """
You are a document indexing assistant for a computer-science literature retrieval system.

Given one text chunk from an arXiv paper, produce structured metadata to improve search.
Do not answer questions about the passage. Do not invent facts not supported by the text.

Return JSON matching the schema:

1. ``predicted_title``: concise title for this chunk (your best guess).
2. ``summary``: exactly two lines summarizing the chunk (use a newline between lines).
3. ``keywords``: deduplicated list of named entities AND other high-signal retrieval terms.
4. ``facts``: list of objects with ``fact`` and ``fact_question``:
   - ``fact``: checkable information need (what must be established), not the answer value.
   - ``fact_question``: searchable query someone would use to retrieve evidence for that need.

Return JSON only.
""".strip()

jina_request_sem = asyncio.Semaphore(2)


class ContextFact(BaseModel):
    """One checkable information need and a searchable query for it."""

    fact: str
    fact_question: str

    @field_validator("fact", "fact_question")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned


class ContextEnrichmentResult(BaseModel):
    """LLM-generated metadata for one arXiv chunk."""

    predicted_title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    facts: list[ContextFact] = Field(default_factory=list)

    @field_validator("predicted_title", "summary")
    @classmethod
    def strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def normalize_keywords(self) -> "ContextEnrichmentResult":
        seen: set[str] = set()
        keywords: list[str] = []
        for item in self.keywords:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                keywords.append(cleaned)
                seen.add(cleaned)
        self.keywords = keywords
        return self


class ChunkPayload(BaseModel):
    """Qdrant point payload (embed and LLM ``text``; ``raw_text`` is unused by the graph)."""

    text: str
    enrichments: dict[str, Any] = Field(default_factory=dict)
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        return self.model_dump()


def build_enriched_text(
    *,
    raw_text: str,
    enrichments: dict[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Build the string embedded at upload time."""

    passage = str(raw_text or "").strip()
    if not passage:
        return ""

    enrichment = enrichments or {}
    if not enrichment:
        return passage

    sections: list[str] = []
    title_text = str(title or "").strip()
    if title_text:
        sections.extend(["Title:", title_text, ""])

    summary = str(enrichment.get("summary") or "").strip()
    if summary:
        sections.extend(["Summary:", summary, ""])

    facts = enrichment.get("facts") or []
    fact_lines = [
        f"- {str(row.get('fact') or '').strip()}"
        for row in facts
        if isinstance(row, dict) and str(row.get("fact") or "").strip()
    ]
    if fact_lines:
        sections.append("Present Facts:")
        sections.extend(fact_lines)
        sections.append("")

    question_lines = [
        f"- {str(row.get('fact_question') or '').strip()}"
        for row in facts
        if isinstance(row, dict) and str(row.get("fact_question") or "").strip()
    ]
    if question_lines:
        sections.append("Sample Query Questions:")
        sections.extend(question_lines)
        sections.append("")

    keywords = enrichment.get("keywords") or []
    keyword_text = ", ".join(
        str(item).strip() for item in keywords if isinstance(item, str) and item.strip()
    )
    if keyword_text:
        sections.extend(["Keywords:", keyword_text, ""])

    sections.extend(["Passage:", passage])
    return "\n".join(sections)


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

    enrichments = element.get("enrichment") or {}
    if not isinstance(enrichments, dict):
        enrichments = {}

    enriched = build_enriched_text(raw_text=raw_text, enrichments=enrichments)
    embed_text = with_prefixed_tables(enriched, table_html)
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
        enrichments=enrichments,
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


async def enrich_element(
    element: dict[str, Any],
    *,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> None:
    """Attach enrichment dict to one chunk element (or null on failure)."""

    passage = str(element.get("text") or "").strip()
    if not passage:
        element["enrichment"] = None
        return

    async with semaphore:
        for attempt in range(1, ENRICH_MAX_RETRIES + 1):
            try:
                response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": passage},
                    ],
                    response_format=ContextEnrichmentResult,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Empty parsed enrichment response")
                element["enrichment"] = parsed.model_dump()
                label = element.get("element_id") or index
                print(f"Enriched {index}/{total} ({label})")
                return
            except Exception as exc:
                if attempt >= ENRICH_MAX_RETRIES:
                    logger.warning(
                        "Enrichment failed for %s: %s",
                        element.get("element_id"),
                        exc,
                    )
                    element["enrichment"] = None
                    print(f"Enriched {index}/{total} ({element.get('element_id')}) — skipped")
                    return
                await asyncio.sleep(0.5 * attempt)


async def enrich_all_elements(file_results: list[dict[str, Any]], config) -> None:
    """Run parallel LLM enrichment for every non-empty chunk element."""

    elements: list[dict[str, Any]] = []
    for file_result in file_results:
        for element in file_result.get("elements") or []:
            if isinstance(element, dict) and str(element.get("text") or "").strip():
                elements.append(element)

    if not elements:
        return

    client = AsyncOpenAI(api_key=config.openai_api_key)
    model = config.ingestion_enrichment_model
    semaphore = asyncio.Semaphore(ENRICH_CONCURRENCY)
    total = len(elements)
    await asyncio.gather(
        *[
            enrich_element(
                element,
                client=client,
                model=model,
                semaphore=semaphore,
                index=index,
                total=total,
            )
            for index, element in enumerate(elements, start=1)
        ]
    )


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
    for file_result in file_results:
        for element in file_result.get("elements") or []:
            if not isinstance(element, dict):
                continue
            row = element_to_chunk(
                file_result,
                element,
                manifest_lookup=manifest_lookup,
            )
            if row is not None:
                chunks.append(row)
    return chunks


async def run_upload(do_enrich: bool) -> None:
    """Load chunks.json, optionally enrich, recreate collection, upsert."""

    config = load_ingestion_config()
    file_results = json.loads(config.chunks_path.read_text(encoding="utf-8"))
    if not isinstance(file_results, list):
        raise ValueError("chunks.json must be a JSON array of file results")

    if do_enrich:
        print(f"Enriching chunks across {len(file_results)} file(s)...")
        await enrich_all_elements(file_results, config)
        config.chunks_path.write_text(
            json.dumps(file_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
    parser = argparse.ArgumentParser(description="Upload arXiv chunks to Qdrant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enrich", action="store_true", help="LLM-enrich chunks before upload")
    group.add_argument("--no-enrich", action="store_true", help="Upload without enrichment")
    args = parser.parse_args()
    asyncio.run(run_upload(do_enrich=args.enrich))


if __name__ == "__main__":
    main()
