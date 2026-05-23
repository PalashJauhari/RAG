"""Qdrant hybrid retriever with per-request strategy selection.

Used by :class:`~graph.graph.RetrievalGraph` and HotpotQA benchmarks. Strategy matrix
(see README): ``fast_retrieval`` (dense + optional MMR), ``keyword`` (BM25 only),
``fast_bm25_retrieval`` (dense + BM25 + RRF), ``fast_bm25_late_interaction_retrieval``
(hybrid candidates + ColBERT re-rank via Jina). Multi-query calls batch OpenAI dense and
Jina ColBERT embeddings once per ``retrieve()``, run Qdrant sub-queries concurrently up to
``retrieval_subquery_max_concurrency``, merge in query order, and deduplicate by point id.
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
        # Limits concurrent Jina HTTP calls across retrieve() invocations.
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
        dense_mmr_limit: int | None = None,
        bm25_limit: int | None = None,
        late_interaction_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval for each query, then merge per-query top-k lists.

        When ``retrieval_subquery_parallel`` is true (default), dense and ColBERT embeddings
        are batched once per call and Qdrant sub-queries run concurrently up to
        ``retrieval_subquery_max_concurrency``. Each sub-query returns up to ``top_k`` hits.
        Results are concatenated in query order with point-id deduplication (first occurrence
        wins). There is no cross-query RRF and no global ``[:top_k]`` cap on the merged list.

        Args:
            queries: One or more retrieval query strings.
            strategy: Per-request tier from complexity or strategy_upgrade.
            top_k: Final per sub-query limit; overrides ``retrieval_top_k`` when set.
            dense_mmr_limit: Dense prefetch output / MMR target; overrides settings when set.
            bm25_limit: BM25 prefetch limit; overrides settings when set.
            late_interaction_limit: RRF pool before ColBERT; overrides settings when set.

        Returns:
            Up to ``top_k * len(queries)`` dicts with ``id``, ``score``, ``rank``, and ``payload``.
        """
        clean_queries = [query.strip() for query in queries if query and query.strip()]
        if not clean_queries:
            return []

        limit = top_k or self.config.retrieval_top_k
        dense_limit = dense_mmr_limit or self.config.retrieval_candidate_dense_mmr
        bm25 = bm25_limit or self.config.retrieval_candidate_bm25
        late_limit = late_interaction_limit or self.config.retrieval_candidate_for_late_interaction

        dense_vectors: list[list[float]] | None = None
        if strategy != "keyword":
            dense_vectors = await self.create_dense_embeddings(clean_queries)

        colbert_vectors: list[list[list[float]]] | None = None
        if strategy == "fast_bm25_late_interaction_retrieval":
            colbert_vectors = await self.create_late_interaction_embeddings(clean_queries)

        if self.config.retrieval_subquery_parallel:
            query_results = await self._retrieve_all_parallel(
                clean_queries,
                strategy=strategy,
                top_k=limit,
                dense_mmr_limit=dense_limit,
                bm25_limit=bm25,
                late_interaction_limit=late_limit,
                dense_vectors=dense_vectors,
                colbert_vectors=colbert_vectors,
            )
        else:
            query_results = await self._retrieve_all_sequential(
                clean_queries,
                strategy=strategy,
                top_k=limit,
                dense_mmr_limit=dense_limit,
                bm25_limit=bm25,
                late_interaction_limit=late_limit,
                dense_vectors=dense_vectors,
                colbert_vectors=colbert_vectors,
            )

        seen_ids: set[str] = set()
        final_docs: list[dict[str, Any]] = []
        for points in query_results:
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

    async def _retrieve_all_sequential(
        self,
        clean_queries: list[str],
        *,
        strategy: RetrievalStrategy,
        top_k: int,
        dense_mmr_limit: int,
        bm25_limit: int,
        late_interaction_limit: int,
        dense_vectors: list[list[float]] | None,
        colbert_vectors: list[list[list[float]]] | None,
    ) -> list[list[Any]]:
        query_results: list[list[Any]] = []
        for index, query in enumerate(clean_queries):
            points = await self._retrieve_one(
                query,
                top_k,
                strategy,
                dense_mmr_limit,
                bm25_limit,
                late_interaction_limit,
                dense_vector=dense_vectors[index] if dense_vectors else None,
                colbert_vector=colbert_vectors[index] if colbert_vectors else None,
            )
            query_results.append(points)
        return query_results

    async def _retrieve_all_parallel(
        self,
        clean_queries: list[str],
        *,
        strategy: RetrievalStrategy,
        top_k: int,
        dense_mmr_limit: int,
        bm25_limit: int,
        late_interaction_limit: int,
        dense_vectors: list[list[float]] | None,
        colbert_vectors: list[list[list[float]]] | None,
    ) -> list[list[Any]]:
        max_concurrency = max(1, self.config.retrieval_subquery_max_concurrency)
        sem = asyncio.Semaphore(max_concurrency)

        async def run_one(index: int, query: str) -> list[Any]:
            async with sem:
                return await self._retrieve_one(
                    query,
                    top_k,
                    strategy,
                    dense_mmr_limit,
                    bm25_limit,
                    late_interaction_limit,
                    dense_vector=dense_vectors[index] if dense_vectors else None,
                    colbert_vector=colbert_vectors[index] if colbert_vectors else None,
                )

        tasks = [run_one(index, query) for index, query in enumerate(clean_queries)]
        return list(await asyncio.gather(*tasks))

    def _dense_query(self, dense_vector: list[float], dense_mmr_limit: int) -> Any:
        """Build dense query vector, optionally wrapped with MMR diversification."""

        if not self.config.use_mmr:
            return dense_vector
        return models.NearestQuery(
            nearest=dense_vector,
            mmr=models.Mmr(
                diversity=self.config.retrieval_mmr_diversity,
                candidates_limit=dense_mmr_limit * 3,
            ),
        )

    async def _retrieve_one(
        self,
        query: str,
        top_k: int,
        strategy: RetrievalStrategy,
        dense_mmr_limit: int,
        bm25_limit: int,
        late_interaction_limit: int,
        *,
        dense_vector: list[float] | None = None,
        colbert_vector: list[list[float]] | None = None,
    ) -> list[Any]:
        """Execute a single-query retrieval pipeline for the given strategy tier."""

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

        if dense_vector is None:
            dense_vector = (await self.create_dense_embeddings([query]))[0]
        dense_query = self._dense_query(dense_vector, dense_mmr_limit)

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

        prefetches = [
            models.Prefetch(
                query=dense_query,
                using=self.config.qdrant_dense_vector_name,
                limit=dense_mmr_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=self.config.qdrant_bm25_model),
                using=self.config.qdrant_bm25_vector_name,
                limit=bm25_limit,
            ),
        ]

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

        if strategy == "fast_bm25_late_interaction_retrieval":
            if colbert_vector is None:
                colbert_vector = (await self.create_late_interaction_embeddings([query]))[0]
            candidate_prefetch = models.Prefetch(
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=late_interaction_limit,
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
