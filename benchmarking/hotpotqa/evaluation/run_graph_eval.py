"""Graph eval step for the HotpotQA benchmark (internal; no CLI).

Called from :func:`~benchmarking.hotpotqa.run_graph_benchmark.run_graph_benchmark`.
Runs :class:`~graph.graph.RetrievalGraph` once per question and writes
``graph_results.json`` and ``graph_run_metadata.json`` under the experiment folder.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from benchmarking.hotpotqa.settings import HotpotQASettings
from graph import RetrievalGraph
from output_validation.final_answer import FinalAnswer


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Aggregate wall-clock latencies into mean, percentiles, and min/max."""

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
    """Capture retrieval-related flags and collection name for reproducibility."""

    return {
        "use_bm25": settings.use_bm25,
        "use_late_interaction": settings.use_late_interaction,
        "use_mmr": settings.use_mmr,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_candidate_dense_mmr": settings.retrieval_candidate_dense_mmr,
        "retrieval_candidate_bm25": settings.retrieval_candidate_bm25,
        "retrieval_candidate_for_late_interaction": settings.retrieval_candidate_for_late_interaction,
        "retrieval_loop_max_retries": settings.retrieval_loop_max_retries,
        "qdrant_collection_name": settings.qdrant_collection_name,
    }


def message_name(message: Any) -> str | None:
    """Return graph node name from ``AIMessage.name`` or serialized message dict."""

    if isinstance(message, AIMessage):
        name = getattr(message, "name", None)
        return str(name).strip() if name else None
    if isinstance(message, dict):
        name = message.get("name")
        if name:
            return str(name).strip()
        data = message.get("data")
        if isinstance(data, dict) and data.get("name"):
            return str(data["name"]).strip()
    return None


def message_content(message: Any) -> str | None:
    """Extract string content from an ``AIMessage`` or a LangGraph/LangChain message dict."""

    raw: Any = None
    if isinstance(message, AIMessage):
        raw = message.content
    elif isinstance(message, dict):
        raw = message.get("content")
        if raw is None:
            data = message.get("data")
            if isinstance(data, dict):
                raw = data.get("content")
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        joined = "\n".join(parts).strip()
        return joined or None
    return str(raw).strip() or None


def final_answer_from_messages(
    messages: list[Any],
    *,
    allowed_names: set[str] | None = None,
) -> FinalAnswer:
    """Parse ``FinalAnswer`` JSON from the first usable AI message (same rules as ``api.main``)."""

    fallback = FinalAnswer(answer="", sources=[], confidence="low")
    for message in messages or []:
        if allowed_names is not None:
            name = message_name(message)
            if name and name not in allowed_names:
                continue
        content = message_content(message)
        if not content:
            continue
        try:
            payload = json.loads(content) if content.lstrip().startswith("{") else content
            if isinstance(payload, dict):
                return FinalAnswer.model_validate(payload)
            return FinalAnswer(answer=str(payload), sources=[], confidence="low")
        except (json.JSONDecodeError, ValueError, TypeError):
            return FinalAnswer(answer=content, sources=[], confidence="low")
    return fallback


def graph_run_response(session_id: str, final_state: dict[str, Any]) -> dict[str, Any]:
    """Normalize graph final state into the same shape as ``api.main.get_api_response``."""

    messages = list(final_state.get("messages") or [])
    answer = final_answer_from_messages(
        list(reversed(messages)),
        allowed_names={"answer_node", "partial_answer_node"},
    )
    if not answer.answer:
        for message in reversed(messages):
            content = message_content(message)
            if content:
                answer = FinalAnswer(answer=content, sources=[], confidence="low")
                break

    return {
        "session_id": session_id,
        "answer": answer.answer,
        "sources": answer.sources,
        "confidence": answer.confidence,
        "retrieved_docs": list(final_state.get("retrieved_documents") or []),
    }


def is_partial_answer(final_state: dict[str, Any]) -> bool:
    """Return True when the graph ended at ``partial_answer_node``."""

    for message in reversed(final_state.get("messages") or []):
        if message_name(message) == "partial_answer_node":
            return True
    return False


def retrieved_contexts_from_state(final_state: dict[str, Any]) -> list[str]:
    """Extract raw passage strings from accumulated ``retrieved_documents``."""

    contexts: list[str] = []
    for row in final_state.get("retrieved_documents") or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


async def run_graph_eval(settings: HotpotQASettings | None = None) -> dict[str, Any]:
    """Run the full Factline graph once per question; record answer, contexts, and latency.

    Contexts come from ``final_state["retrieved_documents"]`` (same corpus passed to
    answer / partial_answer nodes). Graph LLM and Qdrant settings are read from repo root
    ``config.settings`` inside :class:`~graph.graph.RetrievalGraph`.

    Args:
        settings: Benchmark settings; defaults to :class:`HotpotQASettings`.

    Returns:
        Dict with ``results`` list and ``metadata`` written to disk.
    """
    settings = settings or HotpotQASettings()
    graph = RetrievalGraph(InMemorySaver())

    records = json.loads(settings.processed_dataset_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_max_questions > 0:
        records = records[: settings.hotpotqa_max_questions]

    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    try:
        for index, record in enumerate(records, start=1):
            session_id = f"hotpotqa-{record['id']}"
            started = time.perf_counter()
            final_state = await graph.run(session_id, record["question"])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)

            api = graph_run_response(session_id, final_state)
            retrieved_contexts = retrieved_contexts_from_state(final_state)
            retrieved_documents = list(final_state.get("retrieved_documents") or [])

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
                    "response": api.get("answer") or "",
                    "confidence": api.get("confidence"),
                    "sources": api.get("sources") or [],
                    "is_partial": is_partial_answer(final_state),
                    "graph_latency_ms": round(elapsed_ms, 3),
                    "retrieved_contexts": retrieved_contexts,
                    "retrieved_context_ids": [
                        str(row.get("id") or "").strip()
                        for row in retrieved_documents
                        if isinstance(row, dict) and str(row.get("id") or "").strip()
                    ],
                    "retrieved_documents": retrieved_documents,
                    "recall_sufficient": final_state.get("recall_sufficient"),
                    "retrieval_retry_count": final_state.get("retrieval_retry_count", 0),
                }
            )
            print(f"Evaluated {index}/{len(records)} questions ({elapsed_ms:.1f} ms)")
    finally:
        await graph.retriever.qdrant.close()

    partial_count = sum(1 for row in results if row.get("is_partial"))
    latency_stats = latency_summary(latencies_ms)
    metadata = {
        "eval_mode": "graph",
        "experiment_name": settings.hotpotqa_experiment_name,
        "question_count": len(results),
        "partial_answer_count": partial_count,
        "partial_answer_percent": round(100.0 * partial_count / len(results), 2) if results else 0.0,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_loop_max_retries": settings.retrieval_loop_max_retries,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "graph_latency": latency_stats,
        "env": env_snapshot(settings),
    }

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.graph_results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    settings.graph_run_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote graph results to {settings.graph_results_path}")
    print(f"Wrote graph run metadata to {settings.graph_run_metadata_path}")

    return {"results": results, "metadata": metadata}
