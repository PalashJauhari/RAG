from __future__ import annotations

import asyncio
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient, models

from config.settings import Settings
from middleware.llm_client import get_embeddings_client


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
        """
        Public execution entrypoint. Runs retrieval for multiple independent queries concurrently,
        deduplicates the results, and sorts them using Reciprocal Rank Fusion (RRF).
        
        Args:
            queries: A list of standalone semantic search queries (e.g., from query_splitter).
            top_k: The final number of documents to return.
            
        Returns:
            A list of dictionary objects representing the top ranked documents.
        """
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

    async def _retrieve_one(self, query: str, top_k: int) -> list[Any]:
        """
        Executes the internal 3-Stage retrieval pipeline for a single query.
        Stage 1: Dense Retrieval (with MMR 3x over-fetch) & Sparse Retrieval (BM25).
        Stage 2: RRF Fusion of Dense and Sparse results into a candidate pool.
        Stage 3: Late Interaction (ColBERT) re-ranking of the fused candidate pool.
        """
        dense_vector = (await self.create_dense_embeddings([query]))[0]
        dense_query: Any = dense_vector
        if self.config.use_mmr:
            # Stage 1a: Dense Retrieval with Maximal Marginal Relevance (MMR)
            # Fetch 3x candidates internally so MMR has enough room to penalize redundant semantic matches
            dense_query = models.NearestQuery(
                nearest=dense_vector,
                mmr=models.Mmr(
                    diversity=self.config.retrieval_mmr_diversity,
                    candidates_limit=self.config.retrieval_candidate_limit * 3,
                ),
            )

        prefetches = [
            models.Prefetch(
                query=dense_query,
                using=self.config.qdrant_dense_vector_name,
                limit=self.config.retrieval_candidate_limit,
            )
        ]
        
        if self.config.use_bm25:
            # Stage 1b: Sparse Retrieval (BM25 Keyword Search)
            prefetches.append(
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.config.qdrant_bm25_model,
                    ),
                    using=self.config.qdrant_bm25_vector_name,
                    limit=self.config.retrieval_candidate_limit,
                )
            )

        if self.config.use_late_interaction:
            colbert_vector = (await self.create_late_interaction_embeddings([query]))[0]
            candidate_prefetch: Any = prefetches[0]
            
            if self.config.use_bm25:
                # Stage 2: Fuse the Dense and Sparse prefetches via RRF
                # This outputs a combined pool of elite candidates
                candidate_prefetch = models.Prefetch(
                    prefetch=prefetches,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self.config.retrieval_candidate_limit,
                )

            # Stage 3: Late Interaction Re-ranking
            # Perform ColBERT cross-attention re-ranking on the fused candidate pool
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
