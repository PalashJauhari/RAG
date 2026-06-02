"""Upload HotpotQA contexts to Qdrant (optional LLM enrichment)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient, models

from benchmarking.hotpotqa.benchmark_config import load_benchmark_config
from config.settings import settings
from middleware.llm_client import get_llm_client

logger = logging.getLogger(__name__)

UPLOAD_BATCH_SIZE = 16
ENRICH_CONCURRENCY = 10
ENRICH_MAX_RETRIES = 3

ENRICHMENT_SYSTEM_PROMPT = """
You are a document indexing assistant for a retrieval benchmark.

Given one passage (all sentences joined), produce structured metadata to improve search.
Do not answer questions about the passage. Do not invent facts not supported by the text.

Return JSON matching the schema:

1. ``predicted_title``: concise title for this passage (your best guess).
2. ``summary``: exactly two lines summarizing the passage (use a newline between lines).
3. ``keywords``: deduplicated list of named entities (people, places, organizations) AND other
   high-signal retrieval terms (dates, product or policy names, acronyms, exact phrases likely
   in keyword search). Short strings only — no full sentences.
4. ``facts``: list of objects with ``fact`` and ``fact_question``:
   - ``fact``: checkable information need (what must be established), not the answer value.
   - ``fact_question``: searchable query someone would use to retrieve evidence for that need.
     Do not assert the answer in the question.

Rules for facts:
- Only include needs grounded in the passage.
- ``fact_question`` should read like a web or corpus search query, not a chat answer.

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
    """LLM-generated metadata for one HotpotQA context passage."""

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
    """Qdrant point payload for one HotpotQA context."""

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


def context_to_chunk(record: dict[str, Any], context: dict[str, Any]) -> tuple[str, ChunkPayload]:
    """Build stable point id and payload for one context."""

    raw_text = str(context.get("text") or "").strip()
    enrichments = context.get("enrichment") or {}
    if not isinstance(enrichments, dict):
        enrichments = {}

    enriched = build_enriched_text(
        raw_text=raw_text,
        enrichments=enrichments,
        title=str(context.get("title") or ""),
    )
    context_id = str(context["context_id"])
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, context_id))

    payload = ChunkPayload(
        text=enriched,
        enrichments=enrichments,
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


async def enrich_context(
    context: dict[str, Any],
    *,
    llm: Any,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> None:
    """Attach enrichment dict to one context (or null on failure)."""

    passage = str(context.get("text") or "").strip()
    if not passage:
        context["enrichment"] = None
        return

    async with semaphore:
        for attempt in range(1, ENRICH_MAX_RETRIES + 1):
            try:
                result = await llm.ainvoke(
                    [
                        SystemMessage(content=ENRICHMENT_SYSTEM_PROMPT),
                        HumanMessage(content=passage),
                    ]
                )
                parsed: ContextEnrichmentResult = result["parsed"]
                context["enrichment"] = parsed.model_dump()
                print(f"Enriched {index}/{total} ({context.get('context_id')})")
                return
            except Exception as exc:
                if attempt >= ENRICH_MAX_RETRIES:
                    logger.warning(
                        "Enrichment failed for %s: %s",
                        context.get("context_id"),
                        exc,
                    )
                    context["enrichment"] = None
                    print(f"Enriched {index}/{total} ({context.get('context_id')}) — skipped")
                    return
                await asyncio.sleep(0.5 * attempt)


async def enrich_all_contexts(records: list[dict[str, Any]]) -> None:
    """Run parallel LLM enrichment for every non-empty context."""

    contexts: list[dict[str, Any]] = []
    for record in records:
        for context in record.get("contexts") or []:
            if str(context.get("text") or "").strip():
                contexts.append(context)

    if not contexts:
        return

    llm = get_llm_client(
        model=settings.query_decomposition_model,
        output_schema=ContextEnrichmentResult,
        include_raw=True,
    )
    semaphore = asyncio.Semaphore(ENRICH_CONCURRENCY)
    total = len(contexts)
    await asyncio.gather(
        *[
            enrich_context(context, llm=llm, semaphore=semaphore, index=index, total=total)
            for index, context in enumerate(contexts, start=1)
        ]
    )


async def recreate_collection(client: AsyncQdrantClient) -> None:
    """Delete existing collection if present, then create a fresh one."""

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
    chunks: list[tuple[str, ChunkPayload]],
) -> None:
    """Embed payload text and upsert points in batches."""

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
        print(f"Uploaded {min(start + UPLOAD_BATCH_SIZE, len(chunks))}/{len(chunks)} points")


async def run_upload(do_enrich: bool) -> None:
    """Load JSON, optionally enrich, recreate collection, upsert all contexts."""

    config = load_benchmark_config()
    records = json.loads(config.processed_dataset_path.read_text(encoding="utf-8"))

    if do_enrich:
        print(f"Enriching contexts for {len(records)} questions...")
        await enrich_all_contexts(records)
        config.processed_dataset_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        await recreate_collection(client)
        await upsert_chunks(client, chunks)
    finally:
        await client.close()


def main() -> None:
    """CLI entry: upload HotpotQA contexts to Qdrant."""

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Upload HotpotQA contexts to Qdrant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enrich", action="store_true", help="LLM-enrich contexts before upload")
    group.add_argument("--no-enrich", action="store_true", help="Upload without enrichment")
    args = parser.parse_args()
    asyncio.run(run_upload(do_enrich=args.enrich))


if __name__ == "__main__":
    main()
