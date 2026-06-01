"""Retrieval step for the HotpotQA benchmark (internal; no CLI).

Called from :func:`~benchmarking.hotpotqa.run_benchmark.run_benchmark`.
Writes ``retrieval_results.json`` and ``run_metadata.json`` under the experiment folder.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from benchmarking.hotpotqa.settings import HotpotQASettings
from benchmarking.hotpotqa.strategy import validate_strategy_env
from benchmarking.hotpotqa.qdrant_payload import get_metadata, get_raw_text
from output_validation.retrieval_strategy import RetrievalStrategy
from retriever.retriever import Retriever


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(latencies_ms)
    n = len(ordered)

    def percentile(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n)))
        return ordered[idx]

    return {
        "mean_ms": sum(ordered) / n,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def env_snapshot(settings: HotpotQASettings) -> dict[str, Any]:
    return {
        "use_bm25": settings.use_bm25,
        "use_late_interaction": settings.use_late_interaction,
        "use_mmr": settings.use_mmr,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_candidate_dense_mmr": settings.retrieval_candidate_dense_mmr,
        "retrieval_candidate_bm25": settings.retrieval_candidate_bm25,
        "retrieval_candidate_for_late_interaction": settings.retrieval_candidate_for_late_interaction,
        "qdrant_collection_name": settings.qdrant_collection_name,
    }


async def run_retrieval_eval(
    strategy: RetrievalStrategy,
    settings: HotpotQASettings | None = None,
) -> dict[str, Any]:
    """Retrieve once per question; record latency, reference vs retrieved contexts.

    Args:
        strategy: One of the four ``RetrievalStrategy`` literals (same as graph/retriever).
        settings: Benchmark settings; defaults to :class:`HotpotQASettings`.

    Returns:
        Dict with ``results`` list and ``metadata`` written to disk.
    """
    settings = settings or HotpotQASettings()
    validate_strategy_env(strategy, settings)

    retriever = Retriever(settings)
    records = json.loads(settings.processed_dataset_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_max_questions > 0:
        records = records[: settings.hotpotqa_max_questions]

    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    try:
        for index, record in enumerate(records, start=1):
            started = time.perf_counter()
            docs = await retriever.retrieve([record["question"]], strategy=strategy)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)

            reference_contexts = [
                context["text"] for context in record["contexts"] if context["is_supporting"]
            ]
            reference_context_ids = [
                context["context_id"] for context in record["contexts"] if context["is_supporting"]
            ]

            results.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "type": record["type"],
                    "level": record["level"],
                    "reference_answer": record["reference_answer"],
                    "reference_contexts": reference_contexts,
                    "reference_context_ids": reference_context_ids,
                    "retrieval_strategy": strategy,
                    "retrieval_latency_ms": round(elapsed_ms, 3),
                    "retrieved_contexts": [
                        get_raw_text(doc["payload"])
                        for doc in docs
                        if doc.get("payload")
                    ],
                    "retrieved_context_ids": [
                        get_metadata(doc["payload"]).get("context_id")
                        or doc["payload"].get("context_id")
                        for doc in docs
                        if doc.get("payload")
                    ],
                    "retrieved_docs": docs,
                }
            )
            print(f"Evaluated {index}/{len(records)} questions ({elapsed_ms:.1f} ms)")
    finally:
        await retriever.qdrant.close()

    latency_stats = latency_summary(latencies_ms)
    metadata = {
        "retrieval_strategy": strategy,
        "experiment_name": settings.hotpotqa_experiment_name,
        "question_count": len(results),
        "retrieval_top_k": settings.retrieval_top_k,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "retriever_latency": latency_stats,
        "env": env_snapshot(settings),
    }

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.retrieval_results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    settings.run_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote retrieval results to {settings.retrieval_results_path}")
    print(f"Wrote run metadata to {settings.run_metadata_path}")

    return {"results": results, "metadata": metadata}
