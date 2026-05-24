"""Factline LangGraph agent: normalize, decompose facts, retrieve, verify recall, and answer.

High-level flow (see repository README):
    query_normalisation -> fact_decomposition -> query_complexity -> (optional query_splitter) ->
    retrieval -> recall_check -> [answer | gap_fill -> strategy_upgrade -> retrieval] ->
    answer | partial_answer -> clear_turn_trace -> END

Checkpointing: compiled graph uses a LangGraph checkpointer keyed by ``thread_id`` (API
``session_id``). Multi-turn threads persist ``messages`` and ``message_summary``; each
``/run`` resets turn-local scratch while appending a new ``HumanMessage``.

State design: ``messages`` stays lean (user turns + final/partial ``AIMessage`` JSON only).
Intermediate outputs use explicit keys including ``facts``, ``retrieved_documents``,
``retrieved_point_ids``, and ``message_query`` (append-only audit).
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, AsyncIterator, TypedDict

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
from output_validation.recall_check import RecallVerifyResult
from output_validation.message_query_entry import MessageQueryEntry
from output_validation.query_normalisation import QueryNormalisationResult
from output_validation.query_splitter import QuerySplitResult
from output_validation.retrieval_strategy import RetrievalStrategy
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.gap_fill import SYSTEM_PROMPT as GAP_FILL_PROMPT
from prompts.fact_decomposition import SYSTEM_PROMPT as FACT_DECOMPOSITION_PROMPT
from prompts.recall_check import VERIFY_SYSTEM_PROMPT
from prompts.partial_answer import SYSTEM_PROMPT as PARTIAL_ANSWER_PROMPT
from prompts.query_normalisation import SYSTEM_PROMPT as QUERY_NORMALISATION_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import compact_documents_for_llm

class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and per-turn scratch for one LangGraph thread."""

    # --- Conversation (persists across turns on the thread) ---
    # add_messages appends HumanMessage / AIMessage and applies RemoveMessage ops.
    messages: Annotated[list, add_messages]
    message_summary: str  # Rolling summary when context_editing truncation is enabled.

    # --- Turn scratch (reset at each /run invoke) ---
    normalized_query: str
    retrieval_strategy: str  # Tier for retrieval_node; set initially and by deterministic strategy_upgrade.
    active_retrieval_queries: list[str]
    retrieved_documents: Annotated[list[dict[str, Any]], operator.add]  # Accumulates per turn.
    retrieved_point_ids: Annotated[list[str], operator.add]  # Seen Qdrant ids for HasId exclusion.

    # --- Fact scratch (reset each /run; recall_check updates verification in place) ---
    facts: list[dict[str, Any]]
    recall_sufficient: bool  # recall_check → route_after_recall_check

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


def unsupported_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return fact rows still unsupported by the accumulated retrieval corpus."""

    return [row for row in facts if not row.get("verification_status")]


def build_facts_from_decomposition(response: RequiredFactsResult) -> list[dict[str, Any]]:
    """Assign fact_id and empty verification shell after LLM decomposition."""

    return [
        {
            "fact_id": index,
            "fact": item.fact,
            "verification_status": False,
            "verification_report": "",
            "evidence_documents": [],
            "search_queries": [],
            "gap_fill_explanation": "",
        }
        for index, item in enumerate(response.facts, start=1)
    ]


def merge_gap_fill_into_facts(
    facts: list[dict[str, Any]],
    repairs: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge gap-fill repair rows into the unified facts list by ``fact_id``."""

    unsupported = unsupported_facts(facts)
    expected_ids = [row["fact_id"] for row in unsupported]
    repair_ids = [item.fact_id for item in repairs]
    if repair_ids != expected_ids:
        raise ValueError("Gap-fill fact_id values must match unsupported facts by order and id")
    if [item.fact for item in repairs] != fact_texts(unsupported):
        raise ValueError("Gap-fill fact text must match unsupported facts by order and text")

    repair_by_id = {item.fact_id: item for item in repairs}
    updated: list[dict[str, Any]] = []
    queries: list[str] = []
    for row in facts:
        fact_id = row.get("fact_id")
        repair = repair_by_id.get(fact_id)
        if repair is None:
            updated.append(row)
            continue
        updated.append(
            {
                **row,
                "search_queries": list(repair.search_queries),
                "gap_fill_explanation": repair.gap_fill_explanation,
            }
        )
        queries.extend(repair.search_queries)
    return updated, queries


def needs_query_split(facts: list[dict[str, Any]]) -> bool:
    """True when multiple facts require per-fact retrieval queries."""

    return len(facts) > 1


def fact_texts(rows: list[dict[str, Any]]) -> list[str]:
    """Extract non-empty fact strings from list-shaped fact payloads."""

    return [str(row.get("fact") or "").strip() for row in rows if str(row.get("fact") or "").strip()]


def late_interaction_enabled() -> bool:
    """True when ColBERT late-interaction retrieval is configured and allowed."""

    return settings.use_late_interaction and bool(settings.jina_api_key.strip())


def _turn_scratch_reset() -> dict[str, Any]:
    """Default empty values for turn-local state cleared at invoke start and after answer."""

    return {
        "normalized_query": "",
        "retrieval_strategy": "",
        "active_retrieval_queries": [],
        "retrieved_documents": Overwrite(value=[]),
        "retrieved_point_ids": Overwrite(value=[]),
        "facts": [],
        "recall_sufficient": False,
        "retrieval_retry_count": 0,
        "message_query": Overwrite(value=[]),
    }


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
                facts = build_facts_from_decomposition(response)
                node_span.update(
                    output={
                        "normalized_query": normalized_query,
                        "facts": facts,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            facts = build_facts_from_decomposition(response)

        merge: dict[str, Any] = {"facts": facts}
        merge.update(
            trace_row(
                "fact_decomposition",
                "fact_decomposition",
                {"fact_count": len(facts)},
            )
        )
        return merge

    async def query_complexity_node(self, state: RetrievalState) -> dict[str, Any]:
        """Route by fact count and seed queries for the simple path.

        Reads: ``normalized_query`` and ``facts``.
        Writes: ``retrieval_strategy``, ``active_retrieval_queries``.
        Routes via: ``route_after_complexity`` to retrieval or query_splitter.
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        facts = state.get("facts") or []
        fact_count = len(facts)
        needs_split = needs_query_split(facts)
        explanation = (
            f"{fact_count} facts → split retrieval per fact."
            if needs_split
            else "Single fact (or none) → retrieve with normalized query."
        )

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="query_complexity") as node_span:
                node_span.update(
                    output={
                        "needs_split": needs_split,
                        "explanation": explanation,
                        "facts": facts,
                        "fact_count": fact_count,
                    }
                )

        merge = {
            "needs_split": needs_split,
            "retrieval_strategy": "fast_bm25_retrieval",
            "active_retrieval_queries": [normalized_query] if normalized_query else [],
        }
        merge.update(
            trace_row(
                "query_complexity",
                "complexity",
                {"needs_split": needs_split, "fact_count": fact_count},
                notes=explanation,
            )
        )
        return merge

    async def query_splitter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Translate stable facts into focused retrieval strings.

        Writes: ``active_retrieval_queries``. Then fixed edge to ``retrieval``.
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        facts = state.get("facts") or []
        needs_split = needs_query_split(facts)
        llm = get_llm_client(
            model=settings.query_decomposition_model,
            output_schema=QuerySplitResult,
            include_raw=True,
        )
        context = (
            f"## Normalized query\n{normalized_query}\n\n"
            f"## Needs split\n{needs_split}\n\n"
            "## Facts\n"
            f"{json.dumps(facts, ensure_ascii=False)}"
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
                        "facts": facts,
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

        exclude_ids = list(state.get("retrieved_point_ids") or []) or None
        seen_ids = set(state.get("retrieved_point_ids") or [])

        async def run_retrieval() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
            ranked_hits = await self.retriever.retrieve(
                search_queries,
                strategy=strategy,
                top_k=settings.retrieval_top_k,
                dense_mmr_limit=settings.retrieval_candidate_dense_mmr,
                bm25_limit=settings.retrieval_candidate_bm25,
                late_interaction_limit=settings.retrieval_candidate_for_late_interaction,
                exclude_point_ids=exclude_ids,
            )
            compact_document_rows = compact_documents_for_llm(ranked_hits)
            rows_to_add: list[dict[str, Any]] = []
            new_point_ids: list[str] = []
            for row in compact_document_rows:
                point_id = str(row.get("id") or "").strip()
                if not point_id or point_id in seen_ids:
                    continue
                rows_to_add.append(row)
                seen_ids.add(point_id)
                new_point_ids.append(point_id)
            return compact_document_rows, rows_to_add, new_point_ids

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="retrieval") as node_span:
                compact_document_rows, rows_to_add, new_point_ids = await run_retrieval()
                accumulated = list(state.get("retrieved_documents") or [])
                corpus_after = accumulated + rows_to_add
                node_span.update(
                    output={
                        "strategy": strategy,
                        "queries": search_queries,
                        "late_interaction_enabled": late_interaction_enabled(),
                        "exclude_point_id_count": len(exclude_ids or []),
                        "candidate_doc_count": len(compact_document_rows),
                        "rows_to_add": rows_to_add,
                        "rows_to_add_count": len(rows_to_add),
                        "corpus_size_before": len(accumulated),
                        "corpus_size_after": len(corpus_after),
                        "retrieved_documents": corpus_after,
                    }
                )
        else:
            compact_document_rows, rows_to_add, new_point_ids = await run_retrieval()
            accumulated = list(state.get("retrieved_documents") or [])

        merge: dict[str, Any] = {
            "retrieved_documents": rows_to_add,
            "retrieved_point_ids": new_point_ids,
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
                    "rows_to_add_count": len(rows_to_add),
                    "corpus_size_after": len(accumulated) + len(rows_to_add),
                },
            )
        )
        return merge

    # --- Recall / repair nodes ---

    async def recall_check_node(self, state: RetrievalState) -> dict[str, Any]:
        """Verify facts against retrieved passages and update verification in place.

        Writes: ``facts``, ``recall_sufficient``.
        Routes via: ``route_after_recall_check`` (answer | partial_answer | gap_fill).
        """

        normalized = str(state.get("normalized_query") or "").strip()
        facts = list(state.get("facts") or [])
        if not facts:
            raise ValueError("recall_check requires facts from fact_decomposition")
        docs = list(state.get("retrieved_documents") or [])
        doc_lines = [
            f"[{i}] (id={row.get('id')}, score={row.get('score')}) {row.get('text') or ''}"
            for i, row in enumerate(docs, start=1)
        ]
        doc_texts = [str(row.get("text") or "") for row in docs]
        normalized_doc_texts = [
            re.sub(r"\s+", " ", doc_text).strip() for doc_text in doc_texts
        ]

        def _merge_verification_into_facts(
            facts: list[dict[str, Any]],
            verify: RecallVerifyResult,
        ) -> list[dict[str, Any]]:
            expected_texts = fact_texts(facts)
            verified_rows = list(verify.facts)
            verified_texts = [item.fact for item in verified_rows]
            if verified_texts != expected_texts:
                raise ValueError("Recall verification facts must match input facts by order and text")

            updated: list[dict[str, Any]] = []
            for fact_row, verification in zip(facts, verified_rows):
                copied_excerpts = []
                for excerpt in verification.evidence_documents:
                    normalized_excerpt = re.sub(r"\s+", " ", excerpt).strip()
                    if any(excerpt in doc_text for doc_text in doc_texts) or any(
                        normalized_excerpt in doc_text for doc_text in normalized_doc_texts
                    ):
                        copied_excerpts.append(excerpt)
                if verification.verification_status and not copied_excerpts:
                    updated.append(
                        {
                            **fact_row,
                            "verification_status": False,
                            "verification_report": (
                                "Evidence excerpt was not copied verbatim from retrieved documents."
                            ),
                            "evidence_documents": [],
                        }
                    )
                    continue
                updated.append(
                    {
                        **fact_row,
                        "verification_status": verification.verification_status,
                        "verification_report": verification.verification_report,
                        "evidence_documents": copied_excerpts,
                    }
                )
            return updated

        verify_llm = get_llm_client(
            model=settings.recall_check_model,
            output_schema=RecallVerifyResult,
            include_raw=True,
        )
        model = settings.recall_check_model
        verify_human = (
            f"## Normalized query\n{normalized}\n\n"
            "## Facts\n"
            f"{json.dumps(facts, ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{chr(10).join(doc_lines) if doc_lines else '(none)'}"
        )

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="recall_check") as node_span:
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
                updated_facts = _merge_verification_into_facts(facts, ver_result["parsed"])
                unsupported = unsupported_facts(updated_facts)
                recall_sufficient = not unsupported
                node_span.update(
                    output={
                        "facts": updated_facts,
                        "unsupported_facts": unsupported,
                        "recall_sufficient": recall_sufficient,
                        "retrieved_documents": docs,
                    }
                )
        else:
            ver_result = await verify_llm.ainvoke(
                [
                    SystemMessage(content=VERIFY_SYSTEM_PROMPT),
                    HumanMessage(content=verify_human),
                ]
            )
            updated_facts = _merge_verification_into_facts(facts, ver_result["parsed"])
            unsupported = unsupported_facts(updated_facts)
            recall_sufficient = not unsupported

        merge: dict[str, Any] = {
            "facts": updated_facts,
            "recall_sufficient": recall_sufficient,
        }
        merge.update(
            trace_row(
                "recall_check",
                "recall_check",
                {
                    "recall_sufficient": recall_sufficient,
                    "unsupported_facts": unsupported,
                },
            )
        )
        return merge

    async def gap_fill_node(self, state: RetrievalState) -> dict[str, Any]:
        """Generate repair queries per unsupported fact and merge into the unified facts list."""

        unsupported = unsupported_facts(state.get("facts") or [])
        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Unsupported facts\n"
            f"{json.dumps(unsupported, ensure_ascii=False)}\n\n"
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
                updated_facts, queries = merge_gap_fill_into_facts(
                    list(state.get("facts") or []), list(response.facts)
                )
                node_span.update(
                    output={
                        "facts": updated_facts,
                        "active_retrieval_queries": queries,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            updated_facts, queries = merge_gap_fill_into_facts(
                list(state.get("facts") or []), list(response.facts)
            )

        merge: dict[str, Any] = {
            "facts": updated_facts,
            "active_retrieval_queries": queries,
        }
        merge.update(
            trace_row(
                "gap_fill",
                "gap_fill",
                {
                    "queries": queries,
                    "repaired_fact_ids": [row.fact_id for row in response.facts],
                },
                notes="; ".join(row.gap_fill_explanation for row in response.facts),
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

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="strategy_upgrade") as node_span:
                node_span.update(
                    output={
                        "retrieval_strategy": strategy,
                        "retrieval_retry_count": retry_count,
                        "late_threshold": late_threshold,
                        "late_interaction_enabled": late_enabled,
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
                },
                notes=f"Retry {retry_count}: {strategy}.",
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

        Includes per-fact verification in the prompt. Routes to ``clear_turn_trace``.
        """

        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Facts\n"
            f"{json.dumps(state.get('facts') or [], ensure_ascii=False)}\n\n"
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
        """Clear turn-local scratch so the checkpoint is clean after answer/partial_answer."""

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="clear_turn_trace") as node_span:
                node_span.update(output={"cleared": True})
        return _turn_scratch_reset()

    # --- Conditional routing ---

    def route_after_complexity(self, state: RetrievalState) -> str:
        """Route to query_splitter when multiple facts need per-fact queries."""

        if needs_query_split(state.get("facts") or []):
            return "query_splitter"
        return "retrieval"

    def route_after_recall_check(self, state: RetrievalState) -> str:
        """Route after ``recall_check`` (first match wins).

        1. no unsupported facts → ``answer``
        2. retry budget exhausted → ``partial_answer``
        3. unsupported facts remain and retries remain → ``gap_fill``
        """

        if not unsupported_facts(state.get("facts") or []):
            return "answer"
        if state.get("retrieval_retry_count", 0) >= settings.retrieval_loop_max_retries:
            return "partial_answer"
        return "gap_fill"

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
        builder.add_node("gap_fill", self.gap_fill_node)
        builder.add_node("strategy_upgrade", self.strategy_upgrade_node)
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
                "gap_fill": "gap_fill",
            },
        )
        builder.add_edge("gap_fill", "strategy_upgrade")
        builder.add_edge("strategy_upgrade", "retrieval")
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
            **_turn_scratch_reset(),
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
