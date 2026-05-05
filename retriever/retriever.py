from __future__ import annotations

import asyncio
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient, models

from clients.llm_client import get_openai_client
from config.settings import Settings


class Retriever:
    """Runs configured Qdrant retrieval for one or more rewritten queries."""

    def __init__(self, config: Settings):
        self.config = config
        self.openai = get_openai_client(config.openai_api_key)
        self.qdrant = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            cloud_inference=config.use_bm25,
        )

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
        async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
            response = await client.post(
                self.config.jina_multi_vector_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        return [row["embeddings"] for row in response.json()["data"]]

    async def retrieve(self, queries: list[str], top_k: int | None = None) -> list[dict[str, Any]]:
        clean_queries = [query.strip() for query in queries if query and query.strip()]
        if not clean_queries:
            return []

        limit = top_k or self.config.retrieval_top_k
        results = await asyncio.gather(
            *(self._retrieve_one(query, limit) for query in clean_queries)
        )

        docs_by_id: dict[str, dict[str, Any]] = {}
        for points in results:
            for rank, point in enumerate(points):
                point_id = str(point.id)
                payload = point.payload or {}
                rank_score = 1 / (rank + 1)

                if point_id not in docs_by_id:
                    docs_by_id[point_id] = {
                        "id": point_id,
                        "score": point.score,
                        "rank_score": rank_score,
                        "text": payload.get("text")
                        or payload.get("content")
                        or payload.get("page_content")
                        or payload.get("document"),
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
        return docs[:limit]

    async def _retrieve_one(self, query: str, top_k: int) -> list[Any]:
        dense_vector = (await self.create_dense_embeddings([query]))[0]
        dense_query: Any = dense_vector
        if self.config.use_mmr:
            dense_query = models.NearestQuery(
                nearest=dense_vector,
                mmr=models.Mmr(
                    diversity=self.config.retrieval_mmr_diversity,
                    candidates_limit=self.config.retrieval_mmr_candidates_limit,
                ),
            )

        prefetches = [
            models.Prefetch(
                query=dense_query,
                using=self.config.qdrant_dense_vector_name,
                limit=self.config.retrieval_dense_limit,
            )
        ]
        if self.config.use_bm25:
            prefetches.append(
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.config.qdrant_bm25_model,
                    ),
                    using=self.config.qdrant_bm25_vector_name,
                    limit=self.config.retrieval_bm25_limit,
                )
            )

        if self.config.use_late_interaction:
            colbert_vector = (await self.create_late_interaction_embeddings([query]))[0]
            candidate_prefetch: Any = prefetches[0]
            if self.config.use_bm25:
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

        if self.config.use_bm25:
            response = await self.qdrant.query_points(
                collection_name=self.config.qdrant_collection_name,
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return response.points

        response = await self.qdrant.query_points(
            collection_name=self.config.qdrant_collection_name,
            query=dense_query,
            using=self.config.qdrant_dense_vector_name,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return response.points
