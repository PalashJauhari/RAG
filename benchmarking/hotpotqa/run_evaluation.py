"""Run HotpotQA retrieval or graph eval, RAGAS scoring, and benchmark report."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

try:
    from ragas.metrics.collections import AnswerRelevancy as ResponseRelevancy
except ImportError:  # pragma: no cover
    from ragas.metrics.collections import ResponseRelevancy

from benchmarking.hotpotqa.benchmark_config import (
    DEFAULT_RAGAS_MODEL,
    HOTPOTQA_ROOT,
    BenchmarkRunConfig,
    load_benchmark_config,
    parse_pq_list,
    validate_experiment_name,
)
from config.settings import settings
from graph import RetrievalGraph
from output_validation.final_answer import FinalAnswer
from output_validation.retrieval_strategy import RETRIEVAL_STRATEGY_ORDER, RetrievalStrategy
from retriever.retriever import Retriever


def parse_strategy(value: str) -> RetrievalStrategy:
    """Return a validated retrieval strategy literal."""

    key = (value or "").strip()
    if key not in RETRIEVAL_STRATEGY_ORDER:
        allowed = ", ".join(RETRIEVAL_STRATEGY_ORDER)
        raise ValueError(f"Unknown retrieval strategy {value!r}. Choose one of: {allowed}")
    return key  # type: ignore[return-value]


def validate_strategy_env(strategy: RetrievalStrategy) -> None:
    """Ensure root .env supports the requested strategy."""

    if strategy in {"keyword", "fast_bm25_retrieval", "fast_bm25_late_interaction_retrieval"}:
        if not settings.use_bm25:
            raise ValueError(f"Strategy {strategy!r} requires USE_BM25=true in repo root .env")

    if strategy == "fast_bm25_late_interaction_retrieval":
        if not settings.use_late_interaction:
            raise ValueError(
                "Strategy fast_bm25_late_interaction_retrieval requires USE_LATE_INTERACTION=true "
                "in repo root .env"
            )
        if not (settings.jina_api_key or "").strip():
            raise ValueError(
                "Strategy fast_bm25_late_interaction_retrieval requires JINA_API_KEY in repo root .env"
            )

    if strategy == "fast_retrieval" and not (settings.openai_api_key or "").strip():
        raise ValueError("Strategy fast_retrieval requires OPENAI_API_KEY in repo root .env")


def get_raw_text(payload: dict[str, Any]) -> str:
    """Return payload ``text`` (catalog/LLM string). Fall back to ``raw_text`` if empty."""

    text = str(payload.get("text") or "")
    if text.strip():
        return text
    additional = payload.get("additional_metadata")
    if isinstance(additional, dict):
        raw = additional.get("raw_text")
        if raw is not None and str(raw).strip():
            return str(raw)
    return text


def get_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return additional_metadata from a retrieved doc payload."""

    additional = payload.get("additional_metadata")
    return dict(additional) if isinstance(additional, dict) else {}


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Aggregate wall-clock latencies."""

    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_s": 0.0,
            "p50_s": 0.0,
            "p95_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
        }
    ordered = sorted(latencies_ms)
    n = len(ordered)

    def percentile(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n)))
        return ordered[idx]

    mean_ms = sum(ordered) / n
    p50_ms = percentile(0.5)
    p95_ms = percentile(0.95)
    min_ms = ordered[0]
    max_ms = ordered[-1]
    return {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "mean_s": mean_ms / 1000.0,
        "p50_s": p50_ms / 1000.0,
        "p95_s": p95_ms / 1000.0,
        "min_s": min_ms / 1000.0,
        "max_s": max_ms / 1000.0,
    }


def env_snapshot() -> dict[str, Any]:
    """Capture retrieval-related settings for reproducibility."""

    return {
        "use_bm25": settings.use_bm25,
        "use_late_interaction": settings.use_late_interaction,
        "use_mmr": settings.use_mmr,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_candidate_dense_mmr": settings.retrieval_candidate_dense_mmr,
        "retrieval_candidate_bm25": settings.retrieval_candidate_bm25,
        "retrieval_candidate_for_late_interaction": settings.retrieval_candidate_for_late_interaction,
        "retrieval_loop_max_retries": settings.retrieval_loop_max_retries,
        "answer_retry_max": settings.answer_retry_max,
        "qdrant_collection_name": settings.qdrant_collection_name,
        "dense_pq": getattr(settings, "dense_pq", "none"),
    }


def message_name(message: Any) -> str | None:
    """Return graph node name from an AIMessage or serialized dict."""

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
    """Extract string content from a graph message."""

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
    """Parse FinalAnswer JSON from the first usable AI message."""

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
    """Normalize graph final state into API-shaped answer + document_catalog."""

    messages = list(final_state.get("messages") or [])
    answer = final_answer_from_messages(
        list(reversed(messages)),
        allowed_names={"answer_node", "partial_answer_node", "error_answer_node"},
    )
    if not answer.answer:
        for message in reversed(messages):
            content = message_content(message)
            if content:
                answer = FinalAnswer(answer=content, sources=[], confidence="low")
                break

    if "final_sources" in final_state:
        sources = list(final_state.get("final_sources") or [])
    else:
        sources = list(answer.sources)

    return {
        "session_id": session_id,
        "answer": answer.answer,
        "sources": sources,
        "confidence": answer.confidence,
        "document_catalog": dict(final_state.get("document_catalog") or {}),
        "cited_document_ids": list(
            final_state.get("cited_document_ids") or answer.cited_document_ids or []
        ),
        "faithfulness_ok": bool(final_state.get("faithfulness_ok")),
        "answer_mode": str(final_state.get("answer_mode") or ""),
    }


def is_partial_answer(final_state: dict[str, Any]) -> bool:
    """True when the turn used partial_answer (answer_mode or message name)."""

    if str(final_state.get("answer_mode") or "") == "partial":
        return True
    for message in reversed(final_state.get("messages") or []):
        if message_name(message) == "partial_answer_node":
            return True
    return False


def retrieved_contexts_from_state(final_state: dict[str, Any]) -> list[str]:
    """Extract raw passage strings from document_catalog values ({text, source, score})."""

    contexts: list[str] = []
    catalog = final_state.get("document_catalog") or {}
    if not isinstance(catalog, dict):
        return contexts
    for row in catalog.values():
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


async def run_retrieval_eval(
    strategy: RetrievalStrategy,
    config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve once per question and write results + metadata."""

    validate_strategy_env(strategy)
    retriever = Retriever(settings)
    records = json.loads(config.processed_dataset_path.read_text(encoding="utf-8"))
    if config.hotpotqa_max_questions > 0:
        records = records[: config.hotpotqa_max_questions]

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
                }
            )
            print(f"Evaluated {index}/{len(records)} questions ({elapsed_ms:.1f} ms)")
    finally:
        await retriever.qdrant.close()

    metadata = {
        "eval_mode": "retrieval",
        "retrieval_strategy": strategy,
        "experiment_name": config.hotpotqa_experiment_name,
        "dense_pq": config.dense_pq,
        "qdrant_collection_name": config.qdrant_collection_name,
        "question_count": len(results),
        "retrieval_top_k": settings.retrieval_top_k,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latency": latency_summary(latencies_ms),
        "env": env_snapshot(),
    }
    return results, metadata


async def run_graph_eval(config) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run RetrievalGraph once per question and write results + metadata."""

    graph = RetrievalGraph(InMemorySaver())
    records = json.loads(config.processed_dataset_path.read_text(encoding="utf-8"))
    if config.hotpotqa_max_questions > 0:
        records = records[: config.hotpotqa_max_questions]

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
            document_catalog = dict(final_state.get("document_catalog") or {})

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
                    "cited_document_ids": api.get("cited_document_ids") or [],
                    "is_partial": is_partial_answer(final_state),
                    "answer_mode": api.get("answer_mode") or "",
                    "faithfulness_ok": api.get("faithfulness_ok"),
                    "faithfulness_retry_count": int(
                        final_state.get("faithfulness_retry_count") or 0
                    ),
                    "graph_latency_ms": round(elapsed_ms, 3),
                    "retrieved_contexts": retrieved_contexts,
                    "retrieved_context_ids": [
                        str(point_id).strip()
                        for point_id in document_catalog.keys()
                        if str(point_id).strip()
                    ],
                    "document_catalog_size": len(document_catalog),
                    "recall_sufficient": final_state.get("recall_sufficient"),
                    "retrieval_retry_count": final_state.get("retrieval_retry_count", 0),
                }
            )
            print(f"Evaluated {index}/{len(records)} questions ({elapsed_ms:.1f} ms)")
    finally:
        await graph.retriever.qdrant.close()

    partial_count = sum(1 for row in results if row.get("is_partial"))
    metadata = {
        "eval_mode": "graph",
        "experiment_name": config.hotpotqa_experiment_name,
        "dense_pq": config.dense_pq,
        "qdrant_collection_name": config.qdrant_collection_name,
        "question_count": len(results),
        "partial_answer_count": partial_count,
        "partial_answer_percent": round(100.0 * partial_count / len(results), 2) if results else 0.0,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_loop_max_retries": settings.retrieval_loop_max_retries,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latency": latency_summary(latencies_ms),
        "env": env_snapshot(),
    }
    return results, metadata


def apply_eval_collection(config: BenchmarkRunConfig) -> None:
    """Point the shared settings singleton at this PQ collection (graph.py unchanged)."""

    settings.qdrant_collection_name = config.qdrant_collection_name
    settings.dense_pq = config.dense_pq


def metric_float(value: Any) -> float | None:
    """Finite float from a RAGAS result; NaN becomes None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


async def run_ragas(config, mode: str) -> dict[str, Any]:
    """Score results with RAGAS and write ragas_results.json.

    Retrieval and graph: LLM context recall + context precision.
    Graph also: faithfulness, factual correctness, response relevancy (N=3).
    """

    rows = json.loads(config.results_path.read_text(encoding="utf-8"))
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(config.hotpotqa_ragas_model, client=client)
    embeddings = OpenAIEmbeddings(
        client=client,
        model=settings.openai_embedding_model,
    )
    context_recall = ContextRecall(llm=llm)
    context_precision = ContextPrecision(llm=llm)
    faithfulness = Faithfulness(llm=llm) if mode == "graph" else None
    factual_correctness = FactualCorrectness(llm=llm) if mode == "graph" else None
    response_relevancy = (
        ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=3)
        if mode == "graph"
        else None
    )

    scored_rows: list[dict[str, Any]] = []
    skipped_empty_contexts = 0
    skipped_empty_answers = 0

    for index, row in enumerate(rows, start=1):
        question = row["question"]
        reference = row["reference_answer"]
        retrieved_contexts = row.get("retrieved_contexts") or []
        if not retrieved_contexts:
            skipped_empty_contexts += 1
            print(f"Skipped {index}/{len(rows)} (empty contexts)")
            continue

        recall = await context_recall.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        precision = await context_precision.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        scored: dict[str, Any] = {
            "id": row["id"],
            "question": question,
            "type": row.get("type", "unknown"),
            "level": row.get("level", "unknown"),
            "context_recall": metric_float(recall.value),
            "context_precision": metric_float(precision.value),
        }
        if mode == "graph":
            scored["is_partial"] = bool(row.get("is_partial", False))
            response = str(row.get("response") or "").strip()
            if not response:
                skipped_empty_answers += 1
                scored["faithfulness"] = None
                scored["factual_correctness"] = None
                scored["response_relevancy"] = None
            else:
                assert faithfulness is not None
                assert factual_correctness is not None
                assert response_relevancy is not None
                faith = await faithfulness.ascore(
                    user_input=question,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
                factual = await factual_correctness.ascore(
                    response=response,
                    reference=reference,
                )
                relevancy = await response_relevancy.ascore(
                    user_input=question,
                    response=response,
                )
                scored["faithfulness"] = metric_float(faith.value)
                scored["factual_correctness"] = metric_float(factual.value)
                scored["response_relevancy"] = metric_float(relevancy.value)
        scored_rows.append(scored)
        print(f"Scored {index}/{len(rows)} questions")

    if not scored_rows:
        raise ValueError("No results to score")

    summary: dict[str, Any] = {
        "total_scored": len(scored_rows),
        "skipped_empty_contexts": skipped_empty_contexts,
        "mean_context_recall": group_mean(scored_rows, "context_recall"),
        "mean_context_precision": group_mean(scored_rows, "context_precision"),
    }
    if mode == "graph":
        question_count = len(rows)
        partial_count = sum(1 for row in rows if row.get("is_partial"))
        summary["skipped_empty_answers"] = skipped_empty_answers
        summary["mean_faithfulness"] = group_mean(scored_rows, "faithfulness")
        summary["mean_factual_correctness"] = group_mean(scored_rows, "factual_correctness")
        summary["mean_response_relevancy"] = group_mean(scored_rows, "response_relevancy")
        summary["partial_answer_count"] = partial_count
        summary["partial_answer_percent"] = (
            round(100.0 * partial_count / question_count, 2) if question_count else 0.0
        )

    output = {"summary": summary, "rows": scored_rows}
    config.ragas_results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote RAGAS results to {config.ragas_results_path}")
    return output


def format_metric(value: float | int | None, digits: int = 4) -> str:
    """Format a number for markdown tables."""

    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def relative_path(path) -> str:
    """Path relative to hotpotqa root when possible."""

    try:
        return str(path.relative_to(HOTPOTQA_ROOT))
    except ValueError:
        return str(path)


def group_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean of key over rows where value is not None."""

    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def write_benchmark_report(config, mode: str, metadata: dict[str, Any]) -> None:
    """Write benchmark_report.md: quality metrics and latency in seconds."""

    ragas = json.loads(config.ragas_results_path.read_text(encoding="utf-8"))
    title = "HotpotQA Graph Benchmark Report" if mode == "graph" else "HotpotQA Retriever Benchmark Report"
    s = ragas.get("summary") or {}
    question_count = metadata.get("question_count") or 0
    partial_count = metadata.get("partial_answer_count", s.get("partial_answer_count", 0))
    partial_percent = metadata.get("partial_answer_percent")
    if partial_percent is None:
        partial_percent = s.get("partial_answer_percent")
    if partial_percent is None and question_count:
        partial_percent = round(100.0 * float(partial_count) / float(question_count), 2)

    lines = [
        f"# {title}",
        "",
        "## Run summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Experiment | `{config.hotpotqa_experiment_name}` |",
        f"| Dense PQ | `{config.dense_pq}` |",
        f"| Qdrant collection | `{config.qdrant_collection_name}` |",
        f"| Eval mode | `{mode}` |",
    ]

    if mode == "retrieval":
        lines.append(f"| Retrieval strategy | `{metadata.get('retrieval_strategy', '—')}` |")

    lines.extend(
        [
            f"| Questions evaluated | {metadata.get('question_count', '—')} |",
            f"| RETRIEVAL_TOP_K | {metadata.get('retrieval_top_k', '—')} |",
            f"| Timestamp (UTC) | {metadata.get('timestamp_utc', '—')} |",
        ]
    )

    lines.extend(["", "## Quality", ""])
    quality_rows = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Context recall | {format_metric(s.get('mean_context_recall'))} |",
        f"| Context precision | {format_metric(s.get('mean_context_precision'))} |",
        f"| Questions scored | {s.get('total_scored', '—')} |",
        f"| Skipped (empty contexts) | {s.get('skipped_empty_contexts', 0)} |",
    ]
    if mode == "graph":
        quality_rows.extend(
            [
                f"| Faithfulness | {format_metric(s.get('mean_faithfulness'))} |",
                f"| Factual correctness | {format_metric(s.get('mean_factual_correctness'))} |",
                f"| Relevancy | {format_metric(s.get('mean_response_relevancy'))} |",
                f"| Partial answers | {partial_count} ({format_metric(partial_percent, 2)}%) |",
                f"| Skipped (empty answers) | {s.get('skipped_empty_answers', 0)} |",
            ]
        )
    lines.extend(quality_rows)
    lines.append("")

    latency_label = "Graph latency" if mode == "graph" else "Retriever latency"
    lines.extend([f"## {latency_label}", ""])
    latency = metadata.get("latency") or {}
    if latency:
        mean_s = latency.get("mean_s")
        p50_s = latency.get("p50_s")
        p95_s = latency.get("p95_s")
        min_s = latency.get("min_s")
        max_s = latency.get("max_s")
        if mean_s is None and latency.get("mean_ms") is not None:
            mean_s = float(latency["mean_ms"]) / 1000.0
            p50_s = float(latency.get("p50_ms") or 0.0) / 1000.0
            p95_s = float(latency.get("p95_ms") or 0.0) / 1000.0
            min_s = float(latency.get("min_ms") or 0.0) / 1000.0
            max_s = float(latency.get("max_ms") or 0.0) / 1000.0
        lines.extend(
            [
                "| Stat | seconds |",
                "|------|---------|",
                f"| Mean | {format_metric(mean_s, 2)} |",
                f"| p50 | {format_metric(p50_s, 2)} |",
                f"| p95 | {format_metric(p95_s, 2)} |",
                f"| Min | {format_metric(min_s, 2)} |",
                f"| Max | {format_metric(max_s, 2)} |",
                "",
            ]
        )

    if ragas.get("rows"):
        lines.extend(["", "## Breakdown by type and level", ""])
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in ragas["rows"]:
            groups[(row.get("type", "unknown"), row.get("level", "unknown"))].append(row)

        if mode == "graph":
            lines.extend(
                [
                    "| Type | Level | Count | Context recall | Context precision | Faithfulness | Factual correctness | Relevancy |",
                    "|------|-------|-------|----------------|-------------------|--------------|---------------------|-----------|",
                ]
            )
            for (qtype, level), group_rows in sorted(groups.items()):
                n = len(group_rows)
                lines.append(
                    f"| {qtype} | {level} | {n} | "
                    f"{format_metric(group_mean(group_rows, 'context_recall'))} | "
                    f"{format_metric(group_mean(group_rows, 'context_precision'))} | "
                    f"{format_metric(group_mean(group_rows, 'faithfulness'))} | "
                    f"{format_metric(group_mean(group_rows, 'factual_correctness'))} | "
                    f"{format_metric(group_mean(group_rows, 'response_relevancy'))} |"
                )
        else:
            lines.extend(
                [
                    "| Type | Level | Count | Context recall | Context precision |",
                    "|------|-------|-------|----------------|-------------------|",
                ]
            )
            for (qtype, level), group_rows in sorted(groups.items()):
                n = len(group_rows)
                lines.append(
                    f"| {qtype} | {level} | {n} | "
                    f"{format_metric(group_mean(group_rows, 'context_recall'))} | "
                    f"{format_metric(group_mean(group_rows, 'context_precision'))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{relative_path(config.results_path)}`",
            f"- `{relative_path(config.run_metadata_path)}`",
            f"- `{relative_path(config.ragas_results_path)}`",
            f"- `{relative_path(config.benchmark_report_path)}`",
            "",
        ]
    )

    config.benchmark_report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark report to {config.benchmark_report_path}")


def assert_experiment_results_dir_available(config: BenchmarkRunConfig) -> None:
    """Refuse to overwrite an existing experiment results directory."""

    if config.results_dir.exists():
        raise SystemExit(
            f"Experiment {config.hotpotqa_experiment_name!r} already exists at "
            f"{config.results_dir}. Choose a new --experiment-name or remove that folder."
        )


async def run_one_pq_evaluation(
    mode: str,
    strategy: str | None,
    config: BenchmarkRunConfig,
) -> None:
    """Eval → RAGAS → report for one PQ collection."""

    apply_eval_collection(config)
    assert_experiment_results_dir_available(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    if mode == "retrieval":
        if not strategy:
            raise ValueError("--strategy is required when --mode retrieval")
        parsed = parse_strategy(strategy)
        results, metadata = await run_retrieval_eval(parsed, config)
    else:
        results, metadata = await run_graph_eval(config)

    config.results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config.run_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote results to {config.results_path}")

    await run_ragas(config, mode)
    write_benchmark_report(config, mode, metadata)
    print(f"Benchmark complete for PQ={config.dense_pq}.")


async def run_evaluation(
    mode: str,
    strategy: str | None,
    experiment_name: str,
    *,
    collection_base: str,
    dense_pq_values: list[str],
    max_questions: int,
    ragas_model: str,
) -> None:
    """Run eval sequentially for each PQ, saving under data/results/<name>/<pq>/."""

    for dense_pq in dense_pq_values:
        config = load_benchmark_config(
            experiment_name,
            dense_pq=dense_pq,
            collection_base=collection_base,
            max_questions=max_questions,
            ragas_model=ragas_model,
        )
        print(
            f"Starting {mode} eval PQ={dense_pq} collection={config.qdrant_collection_name}"
        )
        await run_one_pq_evaluation(mode, strategy, config)


def main() -> None:
    """CLI entry for HotpotQA evaluation."""

    parser = argparse.ArgumentParser(description="HotpotQA retrieval or graph benchmark")
    parser.add_argument("--mode", required=True, choices=["retrieval", "graph"])
    parser.add_argument(
        "--strategy",
        default=None,
        help="Required for retrieval mode (e.g. fast_bm25_retrieval)",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Unique run id; writes under data/results/<name>/<pq>/",
    )
    parser.add_argument(
        "--collection-base",
        required=True,
        help="Must match upload; collections are {base}_{none|pq8|pq16|pq32}",
    )
    parser.add_argument(
        "--pq",
        nargs="+",
        default=["none"],
        help="PQ modes to eval sequentially (none pq8 pq16 pq32). Default: none",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Cap questions from processed JSON (0 = all).",
    )
    parser.add_argument(
        "--ragas-model",
        default=DEFAULT_RAGAS_MODEL,
        help=(
            f"LLM for RAGAS context recall/precision, faithfulness, factual correctness, and relevancy "
            f"(default: {DEFAULT_RAGAS_MODEL})"
        ),
    )
    args = parser.parse_args()
    validate_experiment_name(args.experiment_name)
    asyncio.run(
        run_evaluation(
            args.mode,
            args.strategy,
            args.experiment_name,
            collection_base=args.collection_base,
            dense_pq_values=parse_pq_list(args.pq),
            max_questions=args.max_questions,
            ragas_model=args.ragas_model,
        )
    )


if __name__ == "__main__":
    main()
