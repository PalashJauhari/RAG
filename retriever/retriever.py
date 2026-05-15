from __future__ import annotations

import asyncio
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient, models

from config.settings import Settings
from middleware.llm_client import get_embeddings_client
from output_validation.retrieval_strategy import RetrievalStrategy


class Retriever:
    """
    Advanced 3-Stage Hybrid Retriever.
    Executes Dense (w/ MMR), Sparse (BM25), and Late Interaction (ColBERT) retrieval 
    against a Qdrant vector database. Supports multi-query fusion using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, config: Settings):
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
    ) -> list[dict[str, Any]]:
        """
        Public execution entrypoint. Runs retrieval for multiple independent queries concurrently,
        deduplicates the results, and sorts them using Reciprocal Rank Fusion (RRF) across queries.

        ``strategy`` selects dense / BM25 / fusion / ColBERT behavior per request (overrides are
        encoded in branches below rather than global flags alone).
        """
        clean_queries = [query.strip() for query in queries if query and query.strip()]
        if not clean_queries:
            return []

        limit = top_k or self.config.retrieval_top_k
        results = await asyncio.gather(
            *(self._retrieve_one(query, limit, strategy) for query in clean_queries)
        )

        docs_by_id: dict[str, dict[str, Any]] = {}
        for points in results:
            for rank, point in enumerate(points):
                point_id = str(point.id)
                payload = point.payload or {}
                
                # Manual Reciprocal Rank Fusion (RRF) across the multiple query result sets
                # Documents that show up high in multiple different queries will get a boosted rank_score
                rank_score = 1 / (rank + 1)

                if point_id not in docs_by_id:
                    docs_by_id[point_id] = {
                        "id": point_id,
                        "score": point.score,
                        "rank_score": rank_score,
                        "payload": payload,
                    }
                else:
                    docs_by_id[point_id]["score"] = max(
                        docs_by_id[point_id]["score"],
                        point.score,
                    )
                    docs_by_id[point_id]["rank_score"] += rank_score

        docs = sorted(
            docs_by_id.values(),
            key=lambda doc: (doc["rank_score"], doc["score"]),
            reverse=True,
        )
        
        final_docs = []
        for i, doc in enumerate(docs[:limit]):
            doc["rank"] = i + 1
            del doc["rank_score"]
            final_docs.append(doc)
            
        return final_docs

    async def _retrieve_one(self, query: str, top_k: int, strategy: RetrievalStrategy) -> list[Any]:
        """Single-query pipeline chosen by ``strategy`` (dense / hybrid / BM25-only / +ColBERT)."""

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
            dense_query = models.NearestQuery(
                nearest=dense_vector,
                mmr=models.Mmr(
                    diversity=self.config.retrieval_mmr_diversity,
                    candidates_limit=self.config.retrieval_candidate_limit * 3,
                ),
            )

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
                limit=self.config.retrieval_candidate_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=self.config.qdrant_bm25_model),
                using=self.config.qdrant_bm25_vector_name,
                limit=self.config.retrieval_candidate_limit,
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
