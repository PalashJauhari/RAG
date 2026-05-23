"""LangGraph retrieval agent: normalize, decompose facts, retrieve, verify recall, and answer.

High-level flow (see repository README):
    query_normalisation -> fact_decomposition -> query_complexity -> (optional query_splitter) -> retrieval ->
    recall_check -> [answer | intent_check -> gap_fill / intent_correction_rewriter ->
    strategy_upgrade -> retrieval] ->
    answer | partial_answer -> clear_turn_trace -> END

Checkpointing: compiled graph uses a LangGraph checkpointer keyed by ``thread_id`` (API
``session_id``). Multi-turn threads persist ``messages`` and ``message_summary``; each
``/run`` resets turn-local scratch (queries, docs, retry counters) while appending a new
``HumanMessage``.

State design: ``messages`` stays lean (user turns + final/partial ``AIMessage`` JSON only).
Intermediate outputs use explicit keys: ``normalized_query``, ``active_retrieval_queries``,
``retrieval_strategy``, ``message_query`` (append-only audit), ``retrieved_documents``,
``required_facts``, and ``verified_facts``.
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, AsyncIterator, NamedTuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command, Overwrite

from config.settings import settings
# Optional long-context compaction (summarize + RemoveMessage); disabled below in normalisation node.
# from middleware.context_editing import truncate_and_summarize
from langfuse import propagate_attributes

from middleware.llm_client import get_llm_client
from observability.langfuse_handler import (
    flush_langfuse,
    get_langfuse_client,
    update_llm_generation,
)
from output_validation.fact_decomposition import RequiredFactsResult
from output_validation.final_answer import FinalAnswer
from output_validation.gap_fill import GapFillResult
from output_validation.intent_check import IntentCheckResult
from output_validation.intent_correction_rewriter import IntentCorrectionRewriteResult
from output_validation.recall_check import RecallVerifyResult
from output_validation.message_query_entry import MessageQueryEntry
from output_validation.query_normalisation import QueryNormalisationResult
from output_validation.query_splitter import QuerySplitResult
from output_validation.retrieval_strategy import RetrievalStrategy
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.gap_fill import SYSTEM_PROMPT as GAP_FILL_PROMPT
from prompts.intent_check import SYSTEM_PROMPT as INTENT_CHECK_PROMPT
from prompts.intent_correction_rewriter import (
    SYSTEM_PROMPT as INTENT_CORRECTION_REWRITER_PROMPT,
)
from prompts.fact_decomposition import SYSTEM_PROMPT as FACT_DECOMPOSITION_PROMPT
from prompts.recall_check import VERIFY_SYSTEM_PROMPT
from prompts.partial_answer import SYSTEM_PROMPT as PARTIAL_ANSWER_PROMPT
from prompts.query_normalisation import SYSTEM_PROMPT as QUERY_NORMALISATION_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import compact_hotqa_documents_for_llm

class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and per-turn scratch for one LangGraph thread."""

    # --- Conversation (persists across turns on the thread) ---
    # add_messages appends HumanMessage / AIMessage and applies RemoveMessage ops.
    messages: Annotated[list, add_messages]
    message_summary: str  # Rolling summary when context_editing truncation is enabled.

    # --- Turn scratch (reset at each /run invoke) ---
    normalized_query: str
    query_complexity: dict[str, Any]  # Last QueryComplexityResult; drives route_after_complexity.
    retrieval_strategy: str  # Tier for retrieval_node; set initially and by deterministic strategy_upgrade.
    active_retrieval_queries: list[str]
    retrieved_documents: Annotated[list[dict[str, Any]], operator.add]  # Accumulates per turn.

    # --- Recall / intent scratch (reset each /run; last-write wins on updates) ---
    required_facts: list[dict[str, Any]]  # Information needs from recall decomposition.
    verified_facts: list[dict[str, Any]]  # Per-fact verification status + excerpts.
    fact_intents: list[dict[str, Any]]  # Per unsupported fact intent alignment.
    # Populated when intent_check finds misaligned facts; fed to intent_correction_rewriter.
    intent_mismatch_details: list[dict[str, Any]]
    recall_sufficient: bool  # recall_check → route_after_recall_check
    intent_aligned: bool  # intent_check → route_after_intent_check

    # --- Audit trace (append-only per turn; cleared by clear_turn_trace_node) ---
    message_query: Annotated[list[dict[str, Any]], operator.add]

    # --- Single retry budget (incremented by deterministic strategy_upgrade on each repair loop) ---
    retrieval_retry_count: int


# --- Trace helpers ---


def trace_row(node: str, kind: str, payload: dict[str, Any], notes: str | None = None) -> dict[str, Any]:
    """Build one ``message_query`` fragment for the operator.add reducer.

    Args:
        node: LangGraph node id (e.g. ``retrieval``).
        kind: Trace category (e.g. ``recall_check``, ``query_prep``).
        payload: JSON-safe detail; kept slim to avoid duplicating live state.
        notes: Optional human-readable summary for logs/UI.

    Returns:
        Dict with a single-element ``message_query`` list to merge into state.
    """

    entry = MessageQueryEntry(node=node, kind=kind, payload=payload, notes=notes)
    return {"message_query": [entry.model_dump()]}


def unsupported_facts_from_verified(verified_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return verification rows still unsupported by the accumulated retrieval corpus."""

    return [row for row in verified_facts if not row.get("verification_status")]


def fact_texts(rows: list[dict[str, Any]]) -> list[str]:
    """Extract non-empty fact strings from list-shaped fact payloads."""

    return [str(row.get("fact") or "").strip() for row in rows if str(row.get("fact") or "").strip()]


def complexity_from_required_facts(required_facts: list[dict[str, Any]]) -> dict[str, str]:
    """Deterministic routing label from stable fact count."""

    fact_count = len(required_facts)
    if fact_count > 1:
        return {
            "complexity": "needs_split",
            "explanation": f"{fact_count} required facts → split retrieval per fact.",
        }
    return {
        "complexity": "simple_query",
        "explanation": "Single required fact (or none) → retrieve with normalized query.",
    }


class RetrievalLimits(NamedTuple):
    """Per-pass retrieval limits after repair-loop scaling."""

    top_k: int
    dense_mmr: int
    bm25: int
    late_interaction: int

    def as_dict(self) -> dict[str, int]:
        return {
            "effective_top_k": self.top_k,
            "effective_dense_mmr": self.dense_mmr,
            "effective_bm25": self.bm25,
            "effective_late_interaction": self.late_interaction,
        }


def effective_retrieval_limits(retry_count: int) -> RetrievalLimits:
    """Scale all retrieval limits by ``1 + retry_count``, each capped by its max setting."""

    multiplier = 1 + max(0, retry_count)
    return RetrievalLimits(
        top_k=min(settings.retrieval_top_k * multiplier, settings.retrieval_top_k_max),
        dense_mmr=min(
            settings.retrieval_candidate_dense_mmr * multiplier,
            settings.retrieval_candidate_dense_mmr_max,
        ),
        bm25=min(
            settings.retrieval_candidate_bm25 * multiplier,
            settings.retrieval_candidate_bm25_max,
        ),
        late_interaction=min(
            settings.retrieval_candidate_for_late_interaction * multiplier,
            settings.retrieval_candidate_for_late_interaction_max,
        ),
    )


def late_interaction_enabled() -> bool:
    """True when ColBERT late-interaction retrieval is configured and allowed."""

    return settings.use_late_interaction and bool(settings.jina_api_key.strip())


def build_node_ai_message(
    *,
    node_name: str,
    payload: dict[str, Any],
    raw: AIMessage | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Persist final structured output as JSON while keeping provider IDs/metadata."""

    # Carry OpenAI/LangChain ids through checkpointing; tag which graph node wrote this turn.
    additional_kwargs = dict(getattr(raw, "additional_kwargs", {}) or {})
    additional_kwargs["node"] = node_name
    if extra_metadata:
        additional_kwargs.update(extra_metadata)

    return AIMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=node_name,
        id=getattr(raw, "id", None),
        additional_kwargs=additional_kwargs,
        response_metadata=dict(getattr(raw, "response_metadata", {}) or {}),
        usage_metadata=getattr(raw, "usage_metadata", None),
    )


# --- Graph builder and nodes ---


class RetrievalGraph:
    """Compiles and runs the retrieval LangGraph with a shared :class:`~retriever.retriever.Retriever`.

    Pass a checkpointer from ``api.main`` lifespan (``InMemorySaver`` or ``AsyncPostgresSaver``).
    """

    def __init__(self, checkpointer: Any, postgres_context: Any | None = None) -> None:
        """Store checkpointer, build compiled graph, and construct retriever.

        Args:
            checkpointer: LangGraph saver for thread checkpoints.
            postgres_context: Async context manager for Postgres saver teardown, if used.
        """
        # postgres_context is only set when using AsyncPostgresSaver so close() can exit the conn ctx.
        self.checkpointer = checkpointer
        self._postgres_context = postgres_context
        self.retriever = Retriever(settings)
        self.graph = self.build_graph()

    # --- Query preparation nodes ---
    # Langfuse (when enabled): node span + nested ``{node}-llm`` generation with token counts.

    async def query_normalisation_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite the latest user message into a standalone query using conversation context.

        Reads: ``messages``, ``message_summary``.
        Writes: ``normalized_query``, optional ``message_summary`` / ``messages`` RemoveMessage ops.
        Routes to: ``query_complexity`` (fixed edge).
        """

        messages = state.get("messages", [])
        summary = state.get("message_summary", "")
        # When enabled: evicts old turns, updates summary, returns RemoveMessage ops for checkpoint.
        # summary, kept_messages, remove_ops = await truncate_and_summarize(messages, summary)
        kept_messages = messages  # No truncation: use full history for normalisation context.
        remove_ops = []  # No RemoveMessage updates applied this step.

        # Last HumanMessage in time order = current user utterance for this turn.
        latest_user_query = ""
        for message in reversed(kept_messages):
            if isinstance(message, HumanMessage):
                latest_user_query = str(message.content)
                break

        context = (
            "## Conversation Summary\n"
            f"{summary or '(none)'}\n\n"
            "## Recent Messages\n"
            f"{messages_to_plain_context(kept_messages)}\n\n"
            "## Latest User Query\n"
            f"{latest_user_query}"
        )
        # Structured LLM output → QueryNormalisationResult (standalone query).
        llm = get_llm_client(
            model=settings.query_normalisation_model,
            output_schema=QueryNormalisationResult,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=QUERY_NORMALISATION_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.query_normalisation_model
        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="query_normalisation") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="query_normalisation-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                normalized_query = response.normalized_query.strip() or latest_user_query.strip()
                node_span.update(output={"normalized_query": normalized_query})
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            normalized_query = response.normalized_query.strip() or latest_user_query.strip()

        # remove_ops shrink checkpoint when truncation is on; message_summary holds rolled-up history.
        merge = {
            "messages": remove_ops,
            "message_summary": summary,
            "normalized_query": normalized_query,
        }
        merge.update(trace_row("query_normalisation", "normalisation", {}))
        return merge

    async def fact_decomposition_node(self, state: RetrievalState) -> dict[str, Any]:
        """Create stable required facts once from the normalized query before retrieval."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        llm = get_llm_client(
            model=settings.query_decomposition_model,
            output_schema=RequiredFactsResult,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=FACT_DECOMPOSITION_PROMPT),
            HumanMessage(content=f"## Normalized query\n{normalized_query}"),
        ]
        model = settings.query_decomposition_model

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="fact_decomposition") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="fact_decomposition-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                required_facts = [item.model_dump() for item in response.facts]
                node_span.update(
                    output={
                        "normalized_query": normalized_query,
                        "required_facts": required_facts,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            required_facts = [item.model_dump() for item in response.facts]

        merge: dict[str, Any] = {"required_facts": required_facts}
        merge.update(
            trace_row(
                "fact_decomposition",
                "fact_decomposition",
                {"required_facts": required_facts},
            )
        )
        return merge

    async def query_complexity_node(self, state: RetrievalState) -> dict[str, Any]:
        """Route by fact count and seed queries for the simple path.

        Reads: ``normalized_query`` and stable ``required_facts``.
        Writes: ``query_complexity``, ``active_retrieval_queries``.
        Routes via: ``route_after_complexity`` to retrieval or query_splitter.
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        required_facts = state.get("required_facts") or []
        output = complexity_from_required_facts(required_facts)
        fact_count = len(required_facts)

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="query_complexity") as node_span:
                node_span.update(
                    output={
                        "complexity": output["complexity"],
                        "explanation": output["explanation"],
                        "required_facts": required_facts,
                        "fact_count": fact_count,
                    }
                )

        merge = {
            "query_complexity": output,
            "retrieval_strategy": "fast_bm25_retrieval",
            "active_retrieval_queries": [normalized_query] if normalized_query else [],
        }
        merge.update(
            trace_row(
                "query_complexity",
                "complexity",
                {"complexity": output["complexity"], "fact_count": fact_count},
                notes=output["explanation"],
            )
        )
        return merge

    async def query_splitter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Translate stable required facts into focused retrieval strings.

        Writes: ``active_retrieval_queries``. Then fixed edge to ``retrieval``.
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        required_facts = state.get("required_facts") or []
        complexity = (state.get("query_complexity") or {}).get("complexity", "")
        llm = get_llm_client(
            model=settings.query_decomposition_model,
            output_schema=QuerySplitResult,
            include_raw=True,
        )
        context = (
            f"## Normalized query\n{normalized_query}\n\n"
            f"## Query complexity\n{complexity}\n\n"
            "## Required facts\n"
            f"{json.dumps(required_facts, ensure_ascii=False)}"
        )
        messages_for_llm = [
            SystemMessage(content=QUERY_SPLITTER_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.query_decomposition_model
        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="query_splitter") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="query_splitter-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                queries = [query.strip() for query in response.queries if query.strip()]
                if not queries and normalized_query:
                    queries = [normalized_query]
                node_span.update(
                    output={
                        "required_facts": required_facts,
                        "active_retrieval_queries": queries,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            queries = [query.strip() for query in response.queries if query.strip()]
            if not queries and normalized_query:
                queries = [normalized_query]

        merge = {"active_retrieval_queries": queries}
        merge.update(trace_row("query_splitter", "query_prep", {"queries": queries}))
        return merge

    # --- Retrieval, recall check, and repair loop ---

    async def retrieval_node(self, state: RetrievalState) -> dict[str, Any]:
        """Retrieve using ``active_retrieval_queries`` and ``retrieval_strategy``; append docs.

        Reads: ``active_retrieval_queries``, ``retrieval_strategy``, ``normalized_query`` (fallback).
        Writes: ``retrieved_documents`` delta rows; LangGraph accumulates them for the turn.
        Routes to: ``recall_check`` (fixed edge).
        """

        raw_strategy = str(state.get("retrieval_strategy") or "").strip()
        # Unknown or empty tier → safe hybrid default for this pass.
        strategy: RetrievalStrategy = raw_strategy if raw_strategy in {"fast_retrieval", "fast_bm25_retrieval", "keyword", "fast_bm25_late_interaction_retrieval"} else "fast_bm25_retrieval"
        if strategy == "fast_bm25_late_interaction_retrieval" and not late_interaction_enabled():
            strategy = "fast_bm25_retrieval"

        search_queries = [
            query.strip()
            for query in state.get("active_retrieval_queries") or []
            if query.strip()
        ]
        if not search_queries:
            fallback = str(state.get("normalized_query") or "").strip()
            search_queries = [fallback] if fallback else []

        retry_count = state.get("retrieval_retry_count", 0)
        limits = effective_retrieval_limits(retry_count)

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="retrieval") as node_span:
                ranked_hits = await self.retriever.retrieve(
                    search_queries,
                    strategy=strategy,
                    top_k=limits.top_k,
                    dense_mmr_limit=limits.dense_mmr,
                    bm25_limit=limits.bm25,
                    late_interaction_limit=limits.late_interaction,
                )
                compact_document_rows = compact_hotqa_documents_for_llm(ranked_hits)
                accumulated = list(state.get("retrieved_documents") or [])
                # Dedup by passage text so retries and multi-query fusion do not duplicate corpus rows.
                seen_texts = {doc.get("text") for doc in accumulated if doc.get("text") is not None}
                rows_to_add: list[dict[str, Any]] = []
                for row in compact_document_rows:
                    text = row.get("text")
                    if text is None or text in seen_texts:
                        continue
                    rows_to_add.append(row)
                    seen_texts.add(text)
                corpus_after = accumulated + rows_to_add
                node_span.update(
                    output={
                        "strategy": strategy,
                        "queries": search_queries,
                        "retrieval_retry_count": retry_count,
                        "late_interaction_enabled": late_interaction_enabled(),
                        **limits.as_dict(),
                        "candidate_doc_count": len(compact_document_rows),
                        "rows_to_add": rows_to_add,
                        "rows_to_add_count": len(rows_to_add),
                        "corpus_size_before": len(accumulated),
                        "corpus_size_after": len(corpus_after),
                        "retrieved_documents": corpus_after,
                    }
                )
        else:
            ranked_hits = await self.retriever.retrieve(
                search_queries,
                strategy=strategy,
                top_k=limits.top_k,
                dense_mmr_limit=limits.dense_mmr,
                bm25_limit=limits.bm25,
                late_interaction_limit=limits.late_interaction,
            )
            compact_document_rows = compact_hotqa_documents_for_llm(ranked_hits)
            accumulated = list(state.get("retrieved_documents") or [])
            seen_texts = {doc.get("text") for doc in accumulated if doc.get("text") is not None}
            rows_to_add = []
            for row in compact_document_rows:
                text = row.get("text")
                if text is None or text in seen_texts:
                    continue
                rows_to_add.append(row)
                seen_texts.add(text)

        merge: dict[str, Any] = {
            "retrieved_documents": rows_to_add,
            "retrieval_strategy": strategy,
            "active_retrieval_queries": search_queries,
        }
        merge.update(
            trace_row(
                "retrieval",
                "retrieval",
                {
                    "queries": search_queries,
                    "strategy": strategy,
                    "retrieval_retry_count": retry_count,
                    **limits.as_dict(),
                    "rows_to_add_count": len(rows_to_add),
                    "corpus_size_after": len(accumulated) + len(rows_to_add),
                },
            )
        )
        return merge

    # --- Recall / intent repair nodes ---
    # All LLM work here is strictly sequential (no asyncio.gather).
    # Repair subgraph: intent_check → gap_fill or intent_correction_rewriter → strategy_upgrade → retrieval.

    async def recall_check_node(self, state: RetrievalState) -> dict[str, Any]:
        """Verify stable required facts against retrieved passages.

        Writes: ``verified_facts``, ``recall_sufficient``.
        Routes via: ``route_after_recall_check`` (answer | partial_answer | intent_check).
        """

        normalized = str(state.get("normalized_query") or "").strip()
        required_facts = list(state.get("required_facts") or [])
        if not required_facts:
            raise ValueError("recall_check requires required_facts from fact_decomposition")
        docs = list(state.get("retrieved_documents") or [])
        doc_lines = [
            f"[{i}] (score={row.get('score')}) {row.get('text') or ''}"
            for i, row in enumerate(docs, start=1)
        ]
        doc_texts = [str(row.get("text") or "") for row in docs]
        normalized_doc_texts = [
            re.sub(r"\s+", " ", doc_text).strip() for doc_text in doc_texts
        ]

        def _validate_verified_facts(
            required_facts: list[dict[str, Any]],
            verify: RecallVerifyResult,
        ) -> list[dict[str, Any]]:
            required_texts = fact_texts(required_facts)
            verified_rows = list(verify.facts)
            verified_texts = [item.fact for item in verified_rows]
            if verified_texts != required_texts:
                raise ValueError("Recall verification facts must match required facts by order and text")

            verified: list[dict[str, Any]] = []
            for verification in verified_rows:
                copied_excerpts = []
                for excerpt in verification.evidence_documents:
                    normalized_excerpt = re.sub(r"\s+", " ", excerpt).strip()
                    if any(excerpt in doc_text for doc_text in doc_texts) or any(
                        normalized_excerpt in doc_text for doc_text in normalized_doc_texts
                    ):
                        copied_excerpts.append(excerpt)
                if verification.verification_status and not copied_excerpts:
                    verified.append(
                        {
                            "fact": verification.fact,
                            "verification_status": False,
                            "verification_report": (
                                "Evidence excerpt was not copied verbatim from retrieved documents."
                            ),
                            "evidence_documents": [],
                        }
                    )
                    continue
                row = verification.model_dump()
                row["evidence_documents"] = copied_excerpts
                verified.append(row)
            return verified

        verify_llm = get_llm_client(
            model=settings.recall_check_model,
            output_schema=RecallVerifyResult,
            include_raw=True,
        )
        model = settings.recall_check_model

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="recall_check") as node_span:
                verify_human = (
                    f"## Normalized query\n{normalized}\n\n"
                    "## Required facts\n"
                    f"{json.dumps(required_facts, ensure_ascii=False)}\n\n"
                    "## Retrieved documents\n"
                    f"{chr(10).join(doc_lines) if doc_lines else '(none)'}"
                )
                with langfuse.start_as_current_observation(
                    as_type="generation", name="recall_check-verify_facts-llm", model=model
                ) as gen_ver:
                    ver_result = await verify_llm.ainvoke(
                        [
                            SystemMessage(content=VERIFY_SYSTEM_PROMPT),
                            HumanMessage(content=verify_human),
                        ]
                    )
                    update_llm_generation(gen_ver, model=model, raw=ver_result.get("raw"))
                verified_facts = _validate_verified_facts(
                    required_facts, ver_result["parsed"]
                )
                unsupported_facts = unsupported_facts_from_verified(verified_facts)
                recall_sufficient = not unsupported_facts
                node_span.update(
                    output={
                        "required_facts": required_facts,
                        "verified_facts": verified_facts,
                        "unsupported_facts": unsupported_facts,
                        "recall_sufficient": recall_sufficient,
                        "retrieved_documents": docs,
                    }
                )
        else:
            verify_human = (
                f"## Normalized query\n{normalized}\n\n"
                "## Required facts\n"
                f"{json.dumps(required_facts, ensure_ascii=False)}\n\n"
                "## Retrieved documents\n"
                f"{chr(10).join(doc_lines) if doc_lines else '(none)'}"
            )
            ver_result = await verify_llm.ainvoke(
                [
                    SystemMessage(content=VERIFY_SYSTEM_PROMPT),
                    HumanMessage(content=verify_human),
                ]
            )
            verified_facts = _validate_verified_facts(
                required_facts, ver_result["parsed"]
            )
            unsupported_facts = unsupported_facts_from_verified(verified_facts)
            recall_sufficient = not unsupported_facts

        merge: dict[str, Any] = {
            "verified_facts": verified_facts,
            "recall_sufficient": recall_sufficient,
        }
        merge.update(
            trace_row(
                "recall_check",
                "recall_check",
                {
                    "recall_sufficient": recall_sufficient,
                    "unsupported_facts": unsupported_facts,
                },
            )
        )
        return merge

    async def intent_check_node(self, state: RetrievalState) -> dict[str, Any]:
        """Decide if active retrieval queries target each unsupported fact."""

        unsupported_facts = unsupported_facts_from_verified(state.get("verified_facts") or [])
        unsupported_texts = fact_texts(unsupported_facts)
        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Unsupported facts\n"
            f"{json.dumps(unsupported_facts, ensure_ascii=False)}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.intent_check_model,
            output_schema=IntentCheckResult,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=INTENT_CHECK_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.intent_check_model

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="intent_check") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="intent_check-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                fact_intents = list(response.fact_intents)
                node_span.update(output={"fact_intents": [item.model_dump() for item in fact_intents]})
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            fact_intents = list(response.fact_intents)

        if [row.fact for row in fact_intents] != unsupported_texts:
            raise ValueError("Intent check facts must match unsupported facts by order and text")

        fact_intents_dump: list[dict[str, Any]] = []
        mismatch_details: list[dict[str, Any]] = []
        for row in fact_intents:
            row_dump = row.model_dump()
            fact_intents_dump.append(row_dump)
            if not row.intent_aligned:
                mismatch_details.append(
                    {
                        "fact": row.fact,
                        "intent_mismatch_details": row.intent_mismatch_details,
                    }
                )

        intent_aligned = not mismatch_details
        merge: dict[str, Any] = {
            "fact_intents": fact_intents_dump,
            "intent_aligned": intent_aligned,
            "intent_mismatch_details": mismatch_details,
        }
        merge.update(
            trace_row(
                "intent_check",
                "intent_check",
                {
                    "intent_aligned": intent_aligned,
                    "misaligned_facts": mismatch_details,
                },
            )
        )
        return merge

    async def gap_fill_node(self, state: RetrievalState) -> dict[str, Any]:
        """Generate three retrieval queries for each unsupported, intent-aligned fact."""

        unsupported_facts = unsupported_facts_from_verified(state.get("verified_facts") or [])
        unsupported_texts = fact_texts(unsupported_facts)
        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Unsupported facts\n"
            f"{json.dumps(unsupported_facts, ensure_ascii=False)}\n\n"
            "## Prior active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.gap_fill_model,
            output_schema=GapFillResult,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=GAP_FILL_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.gap_fill_model

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="gap_fill") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="gap_fill-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                fact_queries = list(response.fact_queries)
                node_span.update(
                    output={
                        "fact_queries": [item.model_dump() for item in fact_queries],
                        "gap_fill_explanation": response.gap_fill_explanation,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            fact_queries = list(response.fact_queries)

        if [row.fact for row in fact_queries] != unsupported_texts:
            raise ValueError("Gap-fill query facts must match unsupported facts by order and text")

        queries: list[str] = []
        for row in fact_queries:
            queries.extend(row.search_queries)

        merge: dict[str, Any] = {"active_retrieval_queries": queries}
        merge.update(
            trace_row(
                "gap_fill",
                "gap_fill",
                {"queries": queries, "facts": [row.fact for row in fact_queries]},
                notes=response.gap_fill_explanation,
            )
        )
        return merge

    async def intent_correction_rewriter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Generate three corrected queries for misaligned unsupported facts."""

        unsupported_facts = unsupported_facts_from_verified(state.get("verified_facts") or [])
        unsupported_texts = fact_texts(unsupported_facts)
        mismatch_details = state.get("intent_mismatch_details") or []
        misaligned_facts = [{"fact": row["fact"]} for row in mismatch_details if row.get("fact")]
        misaligned_texts = fact_texts(misaligned_facts)
        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Misaligned facts\n"
            f"{json.dumps(misaligned_facts, ensure_ascii=False)}\n\n"
            "## Intent mismatch details\n"
            f"{json.dumps(mismatch_details, ensure_ascii=False)}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.intent_correction_rewriter_model,
            output_schema=IntentCorrectionRewriteResult,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=INTENT_CORRECTION_REWRITER_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.intent_correction_rewriter_model

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="intent_correction_rewriter") as node_span:
                with langfuse.start_as_current_observation(
                    as_type="generation", name="intent_correction_rewriter-llm", model=model
                ) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                fact_queries = list(response.fact_queries)
                node_span.update(
                    output={
                        "fact_queries": [item.model_dump() for item in fact_queries],
                        "correction_explanation": response.correction_explanation,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            fact_queries = list(response.fact_queries)

        if [row.fact for row in fact_queries] != misaligned_texts:
            raise ValueError("Intent-correction query facts must match misaligned facts by order and text")

        queries: list[str] = []
        corrected_facts = set()
        for row in fact_queries:
            queries.extend(row.search_queries)
            corrected_facts.add(row.fact)
        # Keep every unsupported fact represented in the next retrieval pass.
        for fact in unsupported_texts:
            if fact not in corrected_facts:
                queries.append(fact)

        merge: dict[str, Any] = {"active_retrieval_queries": queries}
        merge.update(
            trace_row(
                "intent_correction_rewriter",
                "intent_correction",
                {"queries": queries, "misaligned_facts": [row.fact for row in fact_queries]},
                notes=response.correction_explanation,
            )
        )
        return merge

    async def strategy_upgrade_node(self, state: RetrievalState) -> dict[str, Any]:
        """Set the next retrieval tier deterministically and increment retry count."""

        retry_count = state.get("retrieval_retry_count", 0) + 1
        late_threshold = max(0, settings.retrieval_loop_max_retries - 2)
        late_enabled = late_interaction_enabled()
        strategy: RetrievalStrategy = (
            "fast_bm25_late_interaction_retrieval"
            if late_enabled and retry_count >= late_threshold
            else "fast_bm25_retrieval"
        )
        limits = effective_retrieval_limits(retry_count)

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="strategy_upgrade") as node_span:
                node_span.update(
                    output={
                        "retrieval_strategy": strategy,
                        "retrieval_retry_count": retry_count,
                        "late_threshold": late_threshold,
                        "late_interaction_enabled": late_enabled,
                        **limits.as_dict(),
                    }
                )

        merge: dict[str, Any] = {
            "retrieval_strategy": strategy,
            "retrieval_retry_count": retry_count,
        }
        merge.update(
            trace_row(
                "strategy_upgrade",
                "strategy_upgrade",
                {
                    "retrieval_strategy": strategy,
                    "retrieval_retry_count": retry_count,
                    "late_threshold": late_threshold,
                    "late_interaction_enabled": late_enabled,
                    **limits.as_dict(),
                },
                notes=(
                    f"Retry {retry_count}: {strategy}; "
                    f"top_k={limits.top_k}, dense_mmr={limits.dense_mmr}, "
                    f"bm25={limits.bm25}, late={limits.late_interaction}."
                ),
            )
        )
        return merge

    # --- Answer and cleanup ---

    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit grounded ``FinalAnswer`` JSON when recall is sufficient.

        Writes: ``messages`` with ``name=answer_node``. API parses this in ``get_api_response``.
        Routes to: ``clear_turn_trace``.
        """

        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Retrieval strategy\n"
            f"{state.get('retrieval_strategy') or ''}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.final_answer_model,
            output_schema=FinalAnswer,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=FINAL_ANSWER_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.final_answer_model
        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="answer") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="answer-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                raw = result["raw"]
                answer = response.model_dump()
                preview = (answer.get("answer") or "")[:200]
                node_span.update(
                    output={
                        "confidence": answer.get("confidence"),
                        "answer": preview,
                        "retrieved_documents": state.get("retrieved_documents") or [],
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            raw = result["raw"]
            answer = response.model_dump()
        return {
            "messages": [
                build_node_ai_message(node_name="answer_node", payload=answer, raw=raw)
            ]
        }

    async def partial_answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit grounded partial answer when retry budgets are exhausted.

        Includes per-fact verification and retry context in the prompt. Routes to ``clear_turn_trace``.
        """

        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Retrieval strategy\n"
            f"{state.get('retrieval_strategy') or ''}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Required facts\n"
            f"{json.dumps(state.get('required_facts') or [], ensure_ascii=False)}\n\n"
            "## Verified facts\n"
            f"{json.dumps(state.get('verified_facts') or [], ensure_ascii=False)}\n\n"
            "## Unsupported facts\n"
            f"{json.dumps(unsupported_facts_from_verified(state.get('verified_facts') or []), ensure_ascii=False)}\n\n"
            "## Intent mismatch details\n"
            f"{json.dumps(state.get('intent_mismatch_details') or [], ensure_ascii=False)}\n\n"
            "## Retry count\n"
            f"{state.get('retrieval_retry_count', 0)} / {settings.retrieval_loop_max_retries}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.final_answer_model,
            output_schema=FinalAnswer,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=PARTIAL_ANSWER_PROMPT),
            HumanMessage(content=context),
        ]
        model = settings.final_answer_model
        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="partial_answer") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="partial_answer-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                raw = result["raw"]
                answer = response.model_dump()
                preview = (answer.get("answer") or "")[:200]
                node_span.update(
                    output={
                        "confidence": answer.get("confidence"),
                        "answer": preview,
                        "retrieved_documents": state.get("retrieved_documents") or [],
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            raw = result["raw"]
            answer = response.model_dump()
        return {
            "messages": [
                build_node_ai_message(node_name="partial_answer_node", payload=answer, raw=raw)
            ]
        }

    async def clear_turn_trace_node(self, state: RetrievalState) -> dict[str, Any]:
        """Clear append-only audit trace so the next user turn does not leak prior diagnostics.

        Uses ``Overwrite(value=[])`` because ``message_query`` uses ``operator.add`` reducer.
        """

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="clear_turn_trace") as node_span:
                node_span.update(output={"cleared": True})
        return {
            "message_query": Overwrite(value=[]),
            "retrieved_documents": Overwrite(value=[]),
        }

    # --- Conditional routing ---

    def route_after_complexity(self, state: RetrievalState) -> str:
        """Map ``query_complexity.complexity`` to the next graph node name.

        needs_split -> query_splitter
        simple_query (and unknown) -> retrieval (uses seeded active_retrieval_queries)
        """

        complexity = (state.get("query_complexity") or {}).get("complexity")
        if complexity == "needs_split":
            return "query_splitter"
        return "retrieval"

    def route_after_recall_check(self, state: RetrievalState) -> str:
        """Route after ``recall_check`` (first match wins).

        1. no unsupported facts → ``answer``
        2. retry budget exhausted → ``partial_answer``
        3. unsupported facts remain and retries remain → ``intent_check``
        """

        if not unsupported_facts_from_verified(state.get("verified_facts") or []):
            return "answer"
        if state.get("retrieval_retry_count", 0) >= settings.retrieval_loop_max_retries:
            return "partial_answer"
        return "intent_check"

    def route_after_intent_check(self, state: RetrievalState) -> str:
        """Route after ``intent_check``.

        ``intent_aligned`` → evidence-gap path (gap_fill → strategy_upgrade).
        else → query/intent fix (intent_correction_rewriter → strategy_upgrade).
        """

        if state.get("intent_aligned"):
            return "gap_fill"
        return "intent_correction_rewriter"

    # --- Graph wiring ---

    def build_graph(self) -> Any:
        """Compile StateGraph with conditional edges; keys must match router return values."""
        builder = StateGraph(RetrievalState)
        # Internal node ids match strings returned by route_after_* for conditional_edges.
        builder.add_node("query_normalisation", self.query_normalisation_node)
        builder.add_node("fact_decomposition", self.fact_decomposition_node)
        builder.add_node("query_complexity", self.query_complexity_node)
        builder.add_node("query_splitter", self.query_splitter_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("recall_check", self.recall_check_node)
        builder.add_node("intent_check", self.intent_check_node)
        builder.add_node("gap_fill", self.gap_fill_node)
        builder.add_node("strategy_upgrade", self.strategy_upgrade_node)
        builder.add_node("intent_correction_rewriter", self.intent_correction_rewriter_node)
        builder.add_node("answer", self.answer_node)
        builder.add_node("partial_answer", self.partial_answer_node)
        builder.add_node("clear_turn_trace", self.clear_turn_trace_node)

        builder.set_entry_point("query_normalisation")
        builder.add_edge("query_normalisation", "fact_decomposition")
        builder.add_edge("fact_decomposition", "query_complexity")
        # Dict keys must equal route_after_complexity return values (retrieval | query_splitter).
        builder.add_conditional_edges(
            "query_complexity",
            self.route_after_complexity,
            {
                "retrieval": "retrieval",
                "query_splitter": "query_splitter",
            },
        )
        builder.add_edge("query_splitter", "retrieval")
        # --- Post-retrieval: recall gate and repair loops ---
        builder.add_edge("retrieval", "recall_check")
        builder.add_conditional_edges(
            "recall_check",
            self.route_after_recall_check,
            {
                "answer": "answer",
                "partial_answer": "partial_answer",
                "intent_check": "intent_check",
            },
        )
        builder.add_conditional_edges(
            "intent_check",
            self.route_after_intent_check,
            {
                "gap_fill": "gap_fill",
                "intent_correction_rewriter": "intent_correction_rewriter",
            },
        )
        # Intent-aligned repair chain (sequential nodes; no parallel branches).
        builder.add_edge("gap_fill", "strategy_upgrade")
        builder.add_edge("strategy_upgrade", "retrieval")
        # Intent-mismatch correction shares the same deterministic retry/tier step.
        builder.add_edge("intent_correction_rewriter", "strategy_upgrade")
        builder.add_edge("answer", "clear_turn_trace")
        builder.add_edge("partial_answer", "clear_turn_trace")
        builder.add_edge("clear_turn_trace", END)
        # Persists checkpoints keyed by thread_id (session_id from the API).
        return builder.compile(checkpointer=self.checkpointer)

    # --- Public invoke API (called from api.main) ---

    @staticmethod
    def _turn_invoke_input(user_query: str) -> dict[str, Any]:
        """Reset turn-local scratch and append one HumanMessage for a new /run or /run/stream.

        ``message_query: Overwrite([])`` clears prior-turn audit rows at invoke start; the
        terminal ``clear_turn_trace_node`` clears again after answer for checkpoint hygiene.
        """
        return {
            "messages": [HumanMessage(content=user_query)],
            "normalized_query": "",
            "query_complexity": {},
            "retrieval_strategy": "",
            "active_retrieval_queries": [],
            "retrieved_documents": Overwrite(value=[]),
            # Recall / intent repair scratch (see RetrievalState comments).
            "required_facts": [],
            "verified_facts": [],
            "fact_intents": [],
            "intent_mismatch_details": [],
            "recall_sufficient": False,
            "intent_aligned": False,
            "retrieval_retry_count": 0,
            "message_query": Overwrite(value=[]),
        }

    def _invoke_config(self, session_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }

    async def run(
        self,
        session_id: str,
        user_query: str,
    ) -> dict[str, Any]:
        """Run one user turn: append HumanMessage and reset turn-local scratch.

        ``message_query: Overwrite([])`` at invoke start clears prior-turn audit rows.
        ``clear_turn_trace_node`` clears again after answer so the checkpoint stays clean.

        Args:
            session_id: LangGraph ``thread_id``.
            user_query: New user message for this turn.

        Returns:
            Final graph state dict (includes ``messages``, ``retrieved_documents``, etc.).
        """

        config = self._invoke_config(session_id)
        invoke_input = self._turn_invoke_input(user_query)

        if not settings.langfuse_tracing_enabled:
            return await self.graph.ainvoke(invoke_input, config=config)

        langfuse = get_langfuse_client()
        try:
            # Root span: one trace per /run; child node spans nest under this context.
            with langfuse.start_as_current_observation(as_type="span", name="run", metadata={"session_id": session_id}) as root:
                with propagate_attributes(session_id=session_id):
                    return await self.graph.ainvoke(invoke_input, config=config)
        finally:
            flush_langfuse()  # Flush batched observations before returning to the client.

    async def stream_run(
        self,
        session_id: str,
        user_query: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Same input contract as :meth:`run`; yields per-node updates for SSE streaming.

        Yields:
            Dicts keyed by node name (``stream_mode='updates'``).
        """

        config = self._invoke_config(session_id)
        invoke_input = self._turn_invoke_input(user_query)

        if not settings.langfuse_tracing_enabled:
            async for update in self.graph.astream(
                invoke_input,
                config=config,
                stream_mode="updates",
            ):
                yield update
            return

        langfuse = get_langfuse_client()
        try:
            with langfuse.start_as_current_observation(as_type="span", name="stream_run", metadata={"session_id": session_id}) as root:
                with propagate_attributes(session_id=session_id):
                    async for update in self.graph.astream(
                        invoke_input,
                        config=config,
                        stream_mode="updates",
                    ):
                        yield update
        finally:
            flush_langfuse()

    async def resume(
        self,
        session_id: str,
        value: Any,
    ) -> dict[str, Any]:
        """Resume a thread paused by ``interrupt()`` with a human clarification value.

        The current graph completes without pausing; this path is kept for API compatibility.

        Args:
            session_id: LangGraph ``thread_id``.
            value: Human answer passed as ``Command(resume=value)``.

        Returns:
            Final state after resume.
        """
        config = self._invoke_config(session_id)

        if not settings.langfuse_tracing_enabled:
            return await self.graph.ainvoke(Command(resume=value), config=config)

        langfuse = get_langfuse_client()
        try:
            with langfuse.start_as_current_observation(as_type="span", name="resume", metadata={"session_id": session_id}) as root:
                with propagate_attributes(session_id=session_id):
                    return await self.graph.ainvoke(Command(resume=value), config=config)
        finally:
            flush_langfuse()

    def get_state(self, session_id: str) -> Any:
        """Return the latest checkpoint snapshot for ``session_id`` without running the graph."""
        return self.graph.get_state({"configurable": {"thread_id": session_id}})

    async def close(self) -> None:
        """Close Qdrant client and Postgres checkpointer context if configured."""
        await self.retriever.qdrant.close()
        if self._postgres_context is not None:
            await self._postgres_context.__aexit__(None, None, None)
