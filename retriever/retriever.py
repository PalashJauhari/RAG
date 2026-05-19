"""Qdrant hybrid retriever with per-request strategy selection.

Used by :class:`~graph.graph.RetrievalGraph` and HotpotQA benchmarks. Strategy matrix
(see README): ``fast_retrieval`` (dense + optional MMR), ``keyword`` (BM25 only),
``fast_bm25_retrieval`` (dense + BM25 + RRF), ``fast_bm25_late_interaction_retrieval``
(hybrid candidates + ColBERT re-rank via Jina). Multi-query calls keep up to ``top_k`` hits
per sub-query, merge in query order, and deduplicate by point id inside :meth:`Retriever.retrieve`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient, models

from config.settings import Settings
from middleware.llm_client import get_embeddings_client
from output_validation.retrieval_strategy import RetrievalStrategy


class Retriever:
    """Hybrid retriever against a Qdrant collection with dense, sparse, and late interaction."""

    def __init__(self, config: Settings):
        """Wire OpenAI embeddings, Qdrant client, and Jina rate-limit semaphore.

        Args:
            config: App or HotpotQA settings (must expose Qdrant and embedding fields).
        """
        self.config = config
        self.openai = get_embeddings_client(config.openai_api_key)
        self.qdrant = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            cloud_inference=config.use_bm25,
            timeout=config.request_timeout_seconds,
        )
        # Serialized access reduces Jina 429 bursts when multiple sub-queries retrieve in parallel.
        self._jina_request_sem = asyncio.Semaphore(2)

    async def create_dense_embeddings(self, queries: list[str]) -> list[list[float]]:
        """Embed query strings with the configured OpenAI embedding model."""
        response = await self.openai.embeddings.create(
            model=self.config.openai_embedding_model,
            input=queries,
            dimensions=self.config.openai_embedding_dimensions,
        )
        return [item.embedding for item in response.data]

    async def create_late_interaction_embeddings(
        self,
        queries: list[str],
        input_type: str = "query",
    ) -> list[list[list[float]]]:
        """Call Jina multi-vector API for ColBERT-style query embeddings.

        Retries on HTTP 429 with exponential backoff (or ``Retry-After`` when present).
        Concurrent callers share ``_jina_request_sem`` to avoid burst rate limits.

        Args:
            queries: Batch of query texts.
            input_type: Jina ``input_type`` (typically ``query``).

        Returns:
            List of multi-vector embeddings per query.

        Raises:
            ValueError: When ``jina_api_key`` is missing.
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        if not self.config.jina_api_key:
            raise ValueError("JINA_API_KEY is required when USE_LATE_INTERACTION=true")

        payload = {
            "model": self.config.jina_colbert_model,
            "dimensions": self.config.jina_colbert_dimensions,
            "input_type": input_type,
            "embedding_type": "float",
            "input": queries,
        }
        headers = {
            "Authorization": f"Bearer {self.config.jina_api_key}",
            "Content-Type": "application/json",
        }
        max_attempts = 6
        base_delay_seconds = 1.0

        # Semaphore + 429 backoff: multihop queries fan out parallel sub-queries that share Jina quota.
        async with self._jina_request_sem:
            async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                for attempt in range(max_attempts):
                    response = await client.post(
                        self.config.jina_multi_vector_url,
                        headers=headers,
                        json=payload,
                    )
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        retry_after = response.headers.get("retry-after")
                        wait: float
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

    async def retrieve(
        self,
        queries: list[str],
        strategy: RetrievalStrategy,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval for each query concurrently, then merge per-query top-k lists.

        Each sub-query returns up to ``top_k`` (or ``retrieval_top_k``) Qdrant hits. Results are
        concatenated in query order with point-id deduplication (first occurrence wins). There is no
        cross-query RRF and no global ``[:top_k]`` cap on the merged list.

        ``strategy`` selects dense / BM25 / fusion / ColBERT behavior per request (see
        :meth:`_retrieve_one`). Global ``USE_*`` flags still configure the Qdrant client
        and MMR, but the strategy argument chooses which branches execute.

        Args:
            queries: One or more retrieval query strings.
            strategy: Per-request tier from complexity or evaluator.
            top_k: Per sub-query limit; overrides ``retrieval_top_k`` when set.

        Returns:
            Up to ``top_k * len(queries)`` dicts with ``id``, ``score``, ``rank``, and ``payload``.
        """
        clean_queries = [query.strip() for query in queries if query and query.strip()]
        if not clean_queries:
            return []

        limit = top_k or self.config.retrieval_top_k
        results = await asyncio.gather(
            *(self._retrieve_one(query, limit, strategy) for query in clean_queries)
        )

        # --- Merge per-query top-k: query order, dedupe by point id (no cross-query RRF) ---
        seen_ids: set[str] = set()
        final_docs: list[dict[str, Any]] = []
        for points in results:
            for point in points:
                point_id = str(point.id)
                if point_id in seen_ids:
                    continue
                seen_ids.add(point_id)
                final_docs.append(
                    {
                        "id": point_id,
                        "score": point.score,
                        "payload": point.payload or {},
                    }
                )

        for rank, doc in enumerate(final_docs, start=1):
            doc["rank"] = rank

        return final_docs

    async def _retrieve_one(self, query: str, top_k: int, strategy: RetrievalStrategy) -> list[Any]:
        """Execute a single-query retrieval pipeline for the given strategy tier.

        Args:
            query: Raw query text.
            top_k: Maximum points to return for this query.
            strategy: One of the four ``RetrievalStrategy`` literals.

        Returns:
            Qdrant ``ScoredPoint`` list for this query.

        Raises:
            ValueError: Unknown strategy string.
        """

        # --- keyword: BM25-only (no dense embedding) ---
        if strategy == "keyword":
            response = await self.qdrant.query_points(
                collection_name=self.config.qdrant_collection_name,
                query=models.Document(text=query, model=self.config.qdrant_bm25_model),
                using=self.config.qdrant_bm25_vector_name,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return response.points

        dense_vector = (await self.create_dense_embeddings([query]))[0]
        dense_query: Any = dense_vector
        if self.config.use_mmr:
            # MMR diversifies dense candidates before fusion or late interaction.
            dense_query = models.NearestQuery(
                nearest=dense_vector,
                mmr=models.Mmr(
                    diversity=self.config.retrieval_mmr_diversity,
                    candidates_limit=self.config.retrieval_candidate_limit * 3,
                ),
            )

        # --- fast_retrieval: dense (+ optional MMR) only ---
        if strategy == "fast_retrieval":
            response = await self.qdrant.query_points(
                collection_name=self.config.qdrant_collection_name,
                query=dense_query,
                using=self.config.qdrant_dense_vector_name,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return response.points

        # Shared prefetches for hybrid and late-interaction strategies.
        prefetches = [
            models.Prefetch(
                query=dense_query,
                using=self.config.qdrant_dense_vector_name,
                limit=self.config.retrieval_candidate_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=self.config.qdrant_bm25_model),
                using=self.config.qdrant_bm25_vector_name,
                limit=self.config.retrieval_candidate_limit,
            ),
        ]

        # --- fast_bm25_retrieval: dense + BM25 prefetches, RRF fusion ---
        if strategy == "fast_bm25_retrieval":
            response = await self.qdrant.query_points(
                collection_name=self.config.qdrant_collection_name,
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return response.points

        # --- fast_bm25_late_interaction_retrieval: RRF candidates, then ColBERT re-rank ---
        if strategy == "fast_bm25_late_interaction_retrieval":
            colbert_vector = (await self.create_late_interaction_embeddings([query]))[0]
            candidate_prefetch = models.Prefetch(
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=self.config.retrieval_candidate_limit,
            )
            response = await self.qdrant.query_points(
                collection_name=self.config.qdrant_collection_name,
                prefetch=candidate_prefetch,
                query=colbert_vector,
                using=self.config.qdrant_colbert_vector_name,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return response.points

        raise ValueError(f"Unknown retrieval strategy: {strategy!r}")
