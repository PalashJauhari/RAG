"""Factline LangGraph agent: normalize, decompose facts, retrieve, verify recall, and answer.

Overview
--------
This module defines the production RAG **orchestration graph** used by ``api.main``.
Each user turn runs a fixed pipeline of LLM + retrieval nodes. State is checkpointed per
``session_id`` (LangGraph ``thread_id``) so multi-turn chat history survives across
``/run`` calls while per-turn scratch is reset at each invoke start.

High-level flow (see repository README and ``artifacts/langgraph.png``)::

    query_normalisation
        → fact_decomposition
        → query_complexity
        → (optional) query_splitter
        → retrieval
        → recall_check
        → [answer | create_queries_for_unsupported_facts → strategy_upgrade → retrieval]* 
        → answer | partial_answer
        → END

Repair loop (``*``): runs while facts fail recall and ``retrieval_retry_count <
RETRIEVAL_LOOP_MAX_RETRIES``. Each repair pass uses new gap-fill queries and Qdrant
HasId exclusion — not widened top-k.

Checkpointing
-----------
- ``messages`` and ``message_summary`` persist on the thread between turns.
- Turn scratch (``facts``, ``retrieved_documents``, etc.) is wiped via
  :func:`prepare_state_for_next_question` at each ``/run`` / ``/run/stream`` invoke start.
- ``retrieved_documents`` / ``retrieved_point_ids`` use ``operator.add`` reducers during
  a turn so multiple retrieval passes accumulate; ``Overwrite`` clears them on reset.

Unified fact record (in ``state["facts"]``)::

    {
        "fact_id": 1,
        "fact": "Whether X has Y",           # stable after decomposition
        "verification_status": false,        # set by recall_check
        "verification_report": "",
        "evidence_documents": [],              # verbatim excerpts when supported
        "search_queries": [],                # set by create_queries_for_unsupported_facts
        "gap_fill_explanation": "",
    }

Compact retrieved doc (in ``state["retrieved_documents"]`` during a turn)::

    {"id": "<qdrant_point_id>", "score": float, "text": "<raw_text>"}

Module layout
-------------
- :class:`RetrievalState` — TypedDict for checkpointed state keys and reducers.
- Module-level helpers — pure fact/recall utilities (no I/O except async verify helpers).
- :class:`RetrievalGraph` — node implementations, routing, compile, ``run`` / ``stream_run``.
"""

from __future__ import annotations

import asyncio
import json
import operator
from typing import Annotated, Any, AsyncIterator, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# LangGraph: StateGraph builder, END sentinel, and add_messages reducer for messages key.
from langgraph.graph import END, StateGraph, add_messages
# Overwrite replaces list reducer accumulators on reset; Command resumes interrupted graphs.
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
# Pydantic structured-output schemas — one per LLM step in the graph.
from output_validation.fact_decomposition import RequiredFactsResult
from output_validation.final_answer import FinalAnswer
from output_validation.gap_fill import GapFillResult
from output_validation.recall_check import VerifiedFact
from output_validation.query_normalisation import QueryNormalisationResult
from output_validation.query_splitter import QuerySplitResult
from output_validation.retrieval_strategy import RetrievalStrategy
# System prompts (imported as *_PROMPT constants to avoid shadowing node method names).
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.gap_fill import SYSTEM_PROMPT as GAP_FILL_PROMPT
from prompts.fact_decomposition import SYSTEM_PROMPT as FACT_DECOMPOSITION_PROMPT
from prompts.recall_check import VERIFY_SINGLE_FACT_PROMPT
from prompts.partial_answer import SYSTEM_PROMPT as PARTIAL_ANSWER_PROMPT
from prompts.query_normalisation import SYSTEM_PROMPT as QUERY_NORMALISATION_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import compact_documents_for_llm


class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and per-turn scratch for one LangGraph thread.

    Keys marked with reducers behave specially on merge:
    - ``messages``: ``add_messages`` — append HumanMessage / AIMessage; supports RemoveMessage.
    - ``retrieved_documents``: ``operator.add`` — append new doc rows each retrieval pass.
    - ``retrieved_point_ids``: ``operator.add`` — append seen Qdrant ids for HasId exclusion.

    All other keys are replaced by the latest node output unless ``Overwrite`` is used in
    :func:`prepare_state_for_next_question`.
    """

    # --- Conversation (persists across turns on the thread) ---

    messages: Annotated[list, add_messages]
    # Full chat checkpoint: user HumanMessages + final/partial AIMessage JSON only.
    # Intermediate node outputs are NOT stored here (keeps checkpoint small).

    message_summary: str
    # Rolling summary of evicted turns when ``middleware.context_editing`` truncation is
    # enabled. Currently unused (truncation disabled in query_normalisation_node).

    # --- Turn scratch (reset at each /run invoke via prepare_state_for_next_question) ---

    normalized_query: str
    # Standalone query rewritten from the latest user message + thread context.

    retrieval_strategy: str
    # Active Qdrant tier for retrieval_node: e.g. ``fast_bm25_retrieval``,
    # ``fast_bm25_late_interaction_retrieval``. Seeded by query_complexity; upgraded by
    # strategy_upgrade during repair loops.

    active_retrieval_queries: list[str]
    # One or more search strings passed to Retriever.retrieve this pass. Seeded from
    # normalized_query or query_splitter; replaced by create_queries_for_unsupported_facts
    # on repair.

    retrieved_documents: Annotated[list[dict[str, Any]], operator.add]
    # Turn-local corpus of compact docs ``{id, score, text}`` where text is raw_text.
    # Accumulates across retrieval passes until the next invoke's prepare_state_for_next_question.

    retrieved_point_ids: Annotated[list[str], operator.add]
    # All Qdrant point ids seen this turn; fed to HasId must_not on subsequent retrieval.

    # --- Fact scratch (reset each /run; recall_check mutates verification fields) ---

    facts: list[dict[str, Any]]
    # Unified fact records from fact_decomposition; verification updated by recall_check.

    recall_sufficient: bool
    # True when every fact has verification_status=True after recall_check.
    # Drives route_after_recall_check (answer vs repair vs partial_answer).

    # --- Repair loop budget ---

    retrieval_retry_count: int
    # Incremented by strategy_upgrade each repair pass. Compared against
    # settings.retrieval_loop_max_retries to gate partial_answer.


# --- Fact and recall helpers ---
#
# Pure functions (except verify_single_fact / run_verification which call the LLM).
# Called from graph nodes and kept at module level for clarity and testability.


def unsupported_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return fact rows that recall_check marked as not verified.

    Input:
        facts: full ``state["facts"]`` list (supported + unsupported mixed).

    Output:
        Subset where ``verification_status`` is falsy (False, missing, etc.).

    Used by:
        - create_queries_for_unsupported_facts_node (LLM prompt context)
        - add_queries_for_unsupported_facts (filters suggestions by unsupported fact_id)
        - recall_check Langfuse output
    """

    return [row for row in facts if not row.get("verification_status")]


async def verify_single_fact(
    fact_row: dict[str, Any],
    *,
    normalized_query: str,
    docs_block: str,
) -> dict[str, Any]:
    """Run one parallel recall-check LLM call for a single fact row.

    Creates its own structured-output client (``settings.recall_check_model`` +
    ``VerifiedFact`` schema). Does not mutate ``fact_row`` in place — returns a new dict.

    Input:
        fact_row: one unified fact dict with at least ``fact_id`` and ``fact``.
        normalized_query: standalone query for this turn (context for the verifier).
        docs_block: numbered passage block built from ``state["retrieved_documents"]``.

    LLM output (``VerifiedFact`` / ``llm_response["parsed"]``)::

        {
            "fact": "<echo>",
            "verification_status": bool,
            "verification_report": str,
            "evidence_documents": list[str],
        }

    Output:
        Copy of ``fact_row`` with ``verification_status``, ``verification_report``, and
        ``evidence_documents`` overwritten from the verdict. Other keys unchanged
        (``search_queries``, ``gap_fill_explanation``, etc.).

    Raises:
        Propagates LLM / validation errors to the caller (recall_check_node fails the turn).
    """

    # One client per call; parallel invocations share OpenAI rate limiter from get_llm_client.
    verifier = get_llm_client(
        model=settings.recall_check_model,
        output_schema=VerifiedFact,
        include_raw=True,
    )
    human_content = (
        f"## Normalized query\n{normalized_query}\n\n"
        f"## Fact to verify\n"
        f"fact_id: {fact_row.get('fact_id')}\n"
        f"fact: {fact_row.get('fact')}\n\n"
        f"## Retrieved documents\n{docs_block}"
    )
    messages_for_llm = [
        SystemMessage(content=VERIFY_SINGLE_FACT_PROMPT),
        HumanMessage(content=human_content),
    ]
    # Structured invoke returns {"parsed": VerifiedFact, "raw": AIMessage} when include_raw=True.
    llm_response = await verifier.ainvoke(messages_for_llm)
    verdict = llm_response["parsed"]
    # Spread preserves fact_id and any prior repair fields; overwrite verification columns only.
    return {
        **fact_row,
        "verification_status": verdict.verification_status,
        "verification_report": verdict.verification_report,
        "evidence_documents": list(verdict.evidence_documents),
    }


async def run_verification(
    facts: list[dict[str, Any]],
    *,
    normalized_query: str,
    docs_block: str,
) -> list[dict[str, Any]]:
    """Verify all facts in parallel via :func:`verify_single_fact`; preserve order.

    Input:
        facts: full list from ``state["facts"]`` (typically all rows need verification).
        normalized_query: passed through to each verify_single_fact call.
        docs_block: shared numbered corpus string for every fact (full turn corpus).

    Output:
        list[dict] — one updated fact row per input row, **same order** as ``facts``.
        Order follows asyncio.gather task list order, not completion time.

    Note:
        Any single verify_single_fact failure fails the entire gather (no partial results).
        gather preserves task order → output[i] corresponds to facts[i] regardless of finish time.
    """

    return list(
        await asyncio.gather(
            *[
                verify_single_fact(
                    row,
                    normalized_query=normalized_query,
                    docs_block=docs_block,
                )
                for row in facts
            ]
        )
    )


def create_fact_list_with_metadata(response: RequiredFactsResult) -> list[dict[str, Any]]:
    """Build the in-graph fact list from fact_decomposition LLM output.

    Input:
        response: ``RequiredFactsResult`` — ordered list of atomic fact strings from the LLM.

    Output:
        list[dict] — one unified fact record per decomposition item with:
        - ``fact_id``: stable 1-based index for the turn
        - ``fact``: atomic information need (not the answer value)
        - verification / repair fields initialized empty for recall_check and gap-fill

    Does not run verification or retrieval — shape only.
    """

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


def add_queries_for_unsupported_facts(
    facts: list[dict[str, Any]],
    llm_suggested_gap_fill_queries: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge gap-fill LLM suggestions onto unsupported facts; collect flat query list.

    Input:
        facts: **full** fact list (supported + unsupported) from ``state["facts"]``.
        llm_suggested_gap_fill_queries: ``GapFillFact`` rows from ``GapFillResult.facts`` —
            one per unsupported fact, with ``fact_id``, ``search_queries`` (3 strings),
            ``gap_fill_explanation``.

    Output:
        (updated_facts, queries):
        - updated_facts: full list; unsupported rows with a matching LLM suggestion get
          ``search_queries`` and ``gap_fill_explanation``; others unchanged.
        - queries: flat concatenation of suggested ``search_queries`` for matched ids.

    Only suggestions whose ``fact_id`` appears in the current unsupported facts are applied;
    extra or unknown ids from the LLM are ignored (no error).
    """

    # Subset of facts that recall_check marked verification_status=False.
    unsupported = unsupported_facts(facts)
    unsupported_expected_ids = [row["fact_id"] for row in unsupported]

    # Index LLM gap-fill rows by fact_id; ignore ids not in unsupported_expected_ids.
    gap_fill_by_fact_id = {
        item.fact_id: item
        for item in llm_suggested_gap_fill_queries
        if item.fact_id in unsupported_expected_ids
    }

    updated: list[dict[str, Any]] = []
    queries: list[str] = []  # fed to state["active_retrieval_queries"] on the next retrieval pass

    for row in facts:
        fact_id = row.get("fact_id")
        gap_fill = gap_fill_by_fact_id.get(fact_id)

        if gap_fill is None:
            # Supported fact, or unsupported with no matching LLM row: leave row unchanged.
            updated.append(row)
        else:
            # Unsupported fact with gap-fill: attach new search strings for strategy_upgrade.
            updated.append(
                {
                    **row,
                    "search_queries": list(gap_fill.search_queries),
                    "gap_fill_explanation": gap_fill.gap_fill_explanation,
                }
            )
            queries.extend(gap_fill.search_queries)

    return updated, queries


def late_interaction_enabled() -> bool:
    """True when ColBERT late-interaction tier is allowed (flag + Jina API key)."""

    return settings.use_late_interaction and bool(settings.jina_api_key.strip())


def prepare_state_for_next_question(user_query: str) -> dict[str, Any]:
    """Build the state patch for a new user turn before ``graph.ainvoke``.

    Appends one ``HumanMessage`` and wipes all per-turn scratch so a prior turn's facts,
    retrieved corpus, and retry counters do not carry over on the same thread.

    Uses ``Overwrite([])`` on list reducers so accumulated docs/ids are replaced rather
    than appended to stale checkpoint values.

    Args:
        user_query: Raw user message text for this turn.

    Returns:
        State merge dict with ``messages`` plus empty scratch fields
        (``normalized_query``, ``facts``, ``retrieved_documents``, etc.).
    """

    return {
        "messages": [HumanMessage(content=user_query)],
        "normalized_query": "",
        "retrieval_strategy": "",
        "active_retrieval_queries": [],
        "retrieved_documents": Overwrite(value=[]),
        "retrieved_point_ids": Overwrite(value=[]),
        "facts": [],
        "recall_sufficient": False,
        "retrieval_retry_count": 0,
    }


def build_node_ai_message(
    *,
    node_name: str,
    payload: dict[str, Any],
    raw: AIMessage | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Build an AIMessage storing FinalAnswer JSON for checkpoint and API parsing.

    Input:
        node_name: graph node id stored in ``AIMessage.name`` (``answer_node`` or
            ``partial_answer_node``) — API uses this to pick the user-facing message.
        payload: ``FinalAnswer.model_dump()`` dict (answer, sources, confidence).
        raw: original AIMessage from structured LLM invoke (preserves token usage / ids).
        extra_metadata: optional keys merged into ``additional_kwargs``.

    Output:
        AIMessage with JSON string ``content`` and ``additional_kwargs["node"]`` set.
    """

    # Preserve provider metadata from the raw AIMessage for Langfuse / debugging.
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


# --- RetrievalGraph: LangGraph compile, nodes, routing, and public invoke API ---


class RetrievalGraph:
    """Compiles and runs the Factline retrieval LangGraph.

    Responsibilities:
    - Own a shared :class:`~retriever.retriever.Retriever` (Qdrant + embeddings).
    - Register all node callables on a :class:`langgraph.graph.StateGraph`.
    - Expose ``run``, ``stream_run``, ``resume`` for ``api.main``.

    Construction (``api.main`` lifespan):
        graph = RetrievalGraph(InMemorySaver())  # or AsyncPostgresSaver checkpointer

    Each ``/run`` call:
        1. :func:`prepare_state_for_next_question` appends HumanMessage + clears scratch
        2. ``graph.ainvoke`` executes until END
        3. API reads final AIMessage from ``state["messages"]``
    """

    def __init__(self, checkpointer: Any) -> None:
        """Store checkpointer, build compiled graph, and construct retriever.

        Args:
            checkpointer: LangGraph saver (InMemorySaver or AsyncPostgresSaver) keyed by
                ``thread_id`` = API ``session_id``.

        Side effects:
            - Instantiates :class:`~retriever.retriever.Retriever` (Qdrant client, embedders).
            - Calls :meth:`build_graph` and stores compiled graph on ``self.graph``.
        """
        self.checkpointer = checkpointer
        self.retriever = Retriever(settings)
        self.graph = self.build_graph()

    # --- Query preparation nodes ---
    # Flow: normalise user text → decompose facts → seed strategy/queries → (optional) split.
    # Langfuse (when enabled): each node opens a span; LLM calls nest as ``{node}-llm``.

    async def query_normalisation_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite the latest user message into a standalone query using conversation context.

        Purpose:
            Multi-turn questions often use pronouns or omit entities. This node produces
            ``normalized_query`` — a self-contained search/answer string for downstream nodes.

        Reads:
            ``messages`` — full checkpointed chat (HumanMessage + prior AIMessage JSON).
            ``message_summary`` — optional rolled-up history (unused while truncation disabled).

        Writes:
            ``normalized_query`` — from ``QueryNormalisationResult`` or fallback to raw user text.
            ``messages`` — optional ``RemoveMessage`` ops when truncation is enabled (currently []).
            ``message_summary`` — updated summary when truncation is enabled (currently unchanged).

        LLM:
            Model: ``settings.query_normalisation_model``
            Schema: ``QueryNormalisationResult`` (single ``normalized_query`` field).

        Routes to:
            ``fact_decomposition`` (fixed edge).
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
        return merge

    async def fact_decomposition_node(self, state: RetrievalState) -> dict[str, Any]:
        """Decompose the normalized query into atomic required facts (information needs).

        Purpose:
            Split complex questions into verifiable sub-claims before retrieval. Each fact
            becomes one row in ``state["facts"]`` with a stable ``fact_id`` for the turn.

        Reads:
            ``normalized_query``

        Writes:
            ``facts`` — via :func:`create_fact_list_with_metadata` (verification fields empty).

        LLM:
            Model: ``settings.query_decomposition_model``
            Schema: ``RequiredFactsResult`` (ordered list of fact strings).

        Routes to:
            ``query_complexity`` (fixed edge).
        """

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
                facts = create_fact_list_with_metadata(response)
                node_span.update(
                    output={
                        "normalized_query": normalized_query,
                        "facts": facts,
                    }
                )
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            facts = create_fact_list_with_metadata(response)

        return {"facts": facts}

    async def query_complexity_node(self, state: RetrievalState) -> dict[str, Any]:
        """Classify retrieval complexity and seed the first retrieval pass.

        Purpose:
            Decide whether per-fact query splitting is needed (``needs_split``) and initialize
            ``retrieval_strategy`` + ``active_retrieval_queries`` for the first retrieval.

        Reads:
            ``normalized_query``, ``facts``

        Writes:
            ``needs_split`` — ``len(facts) > 1``; also used by Langfuse span output.
            ``retrieval_strategy`` — always ``fast_bm25_retrieval`` on first pass.
            ``active_retrieval_queries`` — ``[normalized_query]`` when query is non-empty.

        Routes via:
            :meth:`route_after_complexity` → ``retrieval`` (0–1 facts) or ``query_splitter`` (>1).
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        facts = state.get("facts") or []
        fact_count = len(facts)
        needs_split = fact_count > 1
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

        return {
            "needs_split": needs_split,
            "retrieval_strategy": "fast_bm25_retrieval",
            "active_retrieval_queries": [normalized_query] if normalized_query else [],
        }

    async def query_splitter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Translate each fact into a focused retrieval query string.

        Purpose:
            When a question decomposes into multiple facts, a single normalized query may
            miss relevant passages for some sub-claims. This node emits one search string
            per fact (or a small aligned set) for the retriever.

        Reads:
            ``normalized_query``, ``facts``

        Writes:
            ``active_retrieval_queries`` — non-empty list; falls back to ``normalized_query``
            if the LLM returns no usable strings.

        LLM:
            Model: ``settings.query_decomposition_model``
            Schema: ``QuerySplitResult`` (``queries: list[str]``).

        Routes to:
            ``retrieval`` (fixed edge).
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        facts = state.get("facts") or []
        llm = get_llm_client(
            model=settings.query_decomposition_model,
            output_schema=QuerySplitResult,
            include_raw=True,
        )
        context = (
            f"## Normalized query\n{normalized_query}\n\n"
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

        return {"active_retrieval_queries": queries}

    # --- Retrieval, recall check, and repair loop ---

    async def retrieval_node(self, state: RetrievalState) -> dict[str, Any]:
        """Retrieve passages from Qdrant using active queries and strategy; append to corpus.

        Purpose:
            Fetch candidate chunks for recall_check and answer generation. During repair
            loops, new queries run against the same collection but exclude already-seen
            point ids (HasId must_not) so each pass adds fresh evidence.

        Reads:
            ``active_retrieval_queries`` — one or more search strings for this pass.
            ``retrieval_strategy`` — Qdrant tier (BM25, hybrid, ColBERT late-interaction).
            ``normalized_query`` — fallback when active queries are empty.
            ``retrieved_point_ids`` — ids to exclude on repair passes.

        Writes:
            ``retrieved_documents`` — **delta** rows appended via ``operator.add`` reducer.
            ``retrieved_point_ids`` — **delta** new ids appended for exclusion next pass.
            ``retrieval_strategy`` — normalized/fallback tier actually used.
            ``active_retrieval_queries`` — echo of queries used (for Langfuse / debugging).

        Retriever:
            ``self.retriever.retrieve(...)`` with ``top_k`` and candidate limits from settings.
            Output slimmed by :func:`compact_documents_for_llm` before state merge.

        Routes to:
            ``recall_check`` (fixed edge).
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

        # HasId exclusion: Qdrant skips chunks already retrieved this turn (repair loops).
        exclude_ids = list(state.get("retrieved_point_ids") or []) or None
        seen_ids = set(state.get("retrieved_point_ids") or [])

        async def run_retrieval() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
            # Retriever may run multiple sub-queries; merge/dedup happens inside Retriever.
            ranked_hits = await self.retriever.retrieve(
                search_queries,
                strategy=strategy,
                top_k=settings.retrieval_top_k,
                dense_mmr_limit=settings.retrieval_candidate_dense_mmr,
                bm25_limit=settings.retrieval_candidate_bm25,
                late_interaction_limit=settings.retrieval_candidate_for_late_interaction,
                exclude_point_ids=exclude_ids,
            )
            # Slim payloads (raw_text, id, score) for LLM nodes; dedup vs checkpoint seen_ids.
            compact_document_rows = compact_documents_for_llm(ranked_hits)
            documents_to_add: list[dict[str, Any]] = []
            new_point_ids: list[str] = []
            for row in compact_document_rows:
                point_id = str(row.get("id") or "").strip()
                if point_id and point_id not in seen_ids:
                    documents_to_add.append(row)
                    seen_ids.add(point_id)
                    new_point_ids.append(point_id)
            return compact_document_rows, documents_to_add, new_point_ids

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="retrieval") as node_span:
                _, documents_to_add, new_point_ids = await run_retrieval()
                already_available_documents = list(state.get("retrieved_documents") or [])
                node_span.update(
                    output={
                        "retrieval_strategy": strategy,
                        "active_retrieval_queries": search_queries,
                        "late_interaction_enabled": late_interaction_enabled(),
                        "documents_to_add": documents_to_add,
                        "retrieved_documents": already_available_documents + documents_to_add,
                    }
                )
        else:
            _, documents_to_add, new_point_ids = await run_retrieval()

        return {
            "retrieved_documents": documents_to_add,
            "retrieved_point_ids": new_point_ids,
            "retrieval_strategy": strategy,
            "active_retrieval_queries": search_queries,
        }

    # --- Recall / repair nodes ---

    async def recall_check_node(self, state: RetrievalState) -> dict[str, Any]:
        """Verify each fact in parallel against retrieved passages; update verification in place.

        Input (from ``state``):
            normalized_query: str — standalone query for this turn.
            facts: list[dict] — unified fact rows from fact_decomposition, e.g.::

                {
                    "fact_id": 1,
                    "fact": "Whether X has Y",
                    "verification_status": false,
                    "verification_report": "",
                    "evidence_documents": [],
                    "search_queries": [],
                    "gap_fill_explanation": "",
                }

            retrieved_documents: list[dict] — compact corpus from retrieval, e.g.::

                {"id": "...", "score": 0.9, "text": "<raw passage>"}

        Output (state merge):
            facts: list[dict] — same rows with verification fields set per fact.
            recall_sufficient: bool — ``True`` only when every fact has
                ``verification_status`` true.

        Routes via: ``route_after_recall_check`` (answer | partial_answer | create_queries_for_unsupported_facts).
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        facts = list(state.get("facts") or [])
        if not facts:
            raise ValueError("recall_check requires facts from fact_decomposition")

        retrieved_docs = list(state.get("retrieved_documents") or [])
        doc_lines = [
            f"[{i}] (id={row.get('id')}, score={row.get('score')}) {row.get('text') or ''}"
            for i, row in enumerate(retrieved_docs, start=1)
        ]
        docs_block = "\n".join(doc_lines) if doc_lines else "(none)"

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="recall_check") as node_span:
                verified_facts = await run_verification(
                    facts,
                    normalized_query=normalized_query,
                    docs_block=docs_block,
                )
                unsupported = unsupported_facts(verified_facts)
                recall_sufficient = all(row["verification_status"] for row in verified_facts)
                node_span.update(
                    output={
                        "facts": verified_facts,
                        "unsupported_facts": unsupported,
                        "unsupported_fact_count": len(unsupported),
                        "recall_sufficient": recall_sufficient,
                        "retrieved_documents": retrieved_docs,
                    }
                )
        else:
            verified_facts = await run_verification(
                facts,
                normalized_query=normalized_query,
                docs_block=docs_block,
            )
            unsupported = unsupported_facts(verified_facts)
            recall_sufficient = all(row["verification_status"] for row in verified_facts)

        return {
            "facts": verified_facts,
            "recall_sufficient": recall_sufficient,
        }

    async def create_queries_for_unsupported_facts_node(self, state: RetrievalState) -> dict[str, Any]:
        """Generate targeted retrieval queries for facts that failed recall verification.

        Purpose:
            When recall_check marks some facts unsupported, this node asks the gap-fill LLM
            to propose new search queries per unsupported fact. Queries are merged onto
            fact rows and flattened into ``active_retrieval_queries`` for the next retrieval.

        Reads:
            ``facts`` (full list), ``normalized_query``, ``active_retrieval_queries``,
            ``retrieved_documents`` (prior corpus for context).

        Writes:
            ``facts`` — unsupported rows get ``search_queries`` + ``gap_fill_explanation``.
            ``active_retrieval_queries`` — flat list of all new query strings.

        LLM:
            Model: ``settings.gap_fill_model``
            Schema: ``GapFillResult`` → merged by :func:`add_queries_for_unsupported_facts`.

        Routes to:
            ``strategy_upgrade`` (fixed edge).
        """

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
            with langfuse.start_as_current_observation(as_type="span", name="create_queries_for_unsupported_facts") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="create_queries_for_unsupported_facts-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                response = result["parsed"]
                updated_facts, queries = add_queries_for_unsupported_facts(
                    list(state.get("facts") or []),
                    list(response.facts),
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
            updated_facts, queries = add_queries_for_unsupported_facts(
                list(state.get("facts") or []),
                list(response.facts),
            )

        return {
            "facts": updated_facts,
            "active_retrieval_queries": queries,
        }

    async def strategy_upgrade_node(self, state: RetrievalState) -> dict[str, Any]:
        """Escalate retrieval tier and increment the repair-loop retry counter.

        Purpose:
            After gap-fill produces new queries, bump ``retrieval_retry_count`` and optionally
            switch from BM25-only to ColBERT late-interaction on later repair attempts.

        Reads:
            ``retrieval_retry_count`` (prior value; incremented by 1).

        Writes:
            ``retrieval_retry_count`` — +1 each repair pass.
            ``retrieval_strategy`` — ``fast_bm25_retrieval`` by default; switches to
                ``fast_bm25_late_interaction_retrieval`` when ColBERT is enabled and
                ``retry_count >= settings.retrieval_loop_max_retries - 2``.

        Routes to:
            ``retrieval`` (fixed edge — starts another retrieval → recall_check cycle).
        """

        retry_count = state.get("retrieval_retry_count", 0) + 1
        # Escalate to ColBERT on the last repair attempts (e.g. retry 2+ when max_retries=3).
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

        return {
            "retrieval_strategy": strategy,
            "retrieval_retry_count": retry_count,
        }

    # --- Answer and cleanup ---

    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Synthesize a grounded final answer when all facts passed recall verification.

        Purpose:
            Produce user-facing ``FinalAnswer`` JSON (answer text, sources, confidence)
            from the accumulated retrieved corpus. Only reached when ``recall_sufficient`` is True.

        Reads:
            ``normalized_query``, ``retrieved_documents``

        Writes:
            ``messages`` — one AIMessage via :func:`build_node_ai_message` with
            ``name="answer_node"``; API parses ``content`` JSON for the SSE ``final`` frame.

        LLM:
            Model: ``settings.final_answer_model``
            Schema: ``FinalAnswer``

        Routes to:
            ``END`` (terminal node).
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

        Includes per-fact verification in the prompt. Routes to ``END``.
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

    # --- Conditional routing ---

    def route_after_complexity(self, state: RetrievalState) -> str:
        """Route to query_splitter when multiple facts need per-fact queries."""

        if len(state.get("facts") or []) > 1:
            return "query_splitter"
        return "retrieval"

    def route_after_recall_check(self, state: RetrievalState) -> str:
        """Route after ``recall_check`` (first match wins).

        1. no unsupported facts → ``answer``
        2. retry budget exhausted → ``partial_answer``
        3. unsupported facts remain and retries remain → ``create_queries_for_unsupported_facts``
        """

        if state.get("recall_sufficient"):
            return "answer"
        if state.get("retrieval_retry_count", 0) >= settings.retrieval_loop_max_retries:
            return "partial_answer"
        return "create_queries_for_unsupported_facts"

    # --- Graph wiring ---

    def build_graph(self) -> Any:
        """Wire nodes and compile the LangGraph with a session checkpointer.

        Repair loop: recall_check → create_queries_for_unsupported_facts → strategy_upgrade → retrieval → recall_check.
        Success exits: recall_check → answer → END.
        Budget exhausted: recall_check → partial_answer → END.
        """
        builder = StateGraph(RetrievalState)
        # Node names must match strings returned by route_after_* for conditional_edges.
        builder.add_node("query_normalisation", self.query_normalisation_node)
        builder.add_node("fact_decomposition", self.fact_decomposition_node)
        builder.add_node("query_complexity", self.query_complexity_node)
        builder.add_node("query_splitter", self.query_splitter_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("recall_check", self.recall_check_node)
        builder.add_node(
            "create_queries_for_unsupported_facts",
            self.create_queries_for_unsupported_facts_node,
        )
        builder.add_node("strategy_upgrade", self.strategy_upgrade_node)
        builder.add_node("answer", self.answer_node)
        builder.add_node("partial_answer", self.partial_answer_node)

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
        # Post-retrieval: verify corpus covers facts; repair or answer.
        builder.add_edge("retrieval", "recall_check")
        builder.add_conditional_edges(
            "recall_check",
            self.route_after_recall_check,
            {
                "answer": "answer",
                "partial_answer": "partial_answer",
                "create_queries_for_unsupported_facts": "create_queries_for_unsupported_facts",
            },
        )
        builder.add_edge("create_queries_for_unsupported_facts", "strategy_upgrade")
        builder.add_edge("strategy_upgrade", "retrieval")
        builder.add_edge("answer", END)
        builder.add_edge("partial_answer", END)
        # Persists checkpoints keyed by thread_id (session_id from the API).
        return builder.compile(checkpointer=self.checkpointer)

    # --- Public invoke API (called from api.main) ---

    def invoke_config(self, session_id: str) -> dict[str, Any]:
        """LangGraph run config: thread checkpoint key and safety limits."""
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

        Args:
            session_id: LangGraph ``thread_id``.
            user_query: New user message for this turn.

        Returns:
            Final graph state dict (includes ``messages``, ``retrieved_documents``, etc.).
        """

        config = self.invoke_config(session_id)
        invoke_input = prepare_state_for_next_question(user_query)

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

        config = self.invoke_config(session_id)
        invoke_input = prepare_state_for_next_question(user_query)

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
        config = self.invoke_config(session_id)

        if not settings.langfuse_tracing_enabled:
            return await self.graph.ainvoke(Command(resume=value), config=config)

        langfuse = get_langfuse_client()
        try:
            with langfuse.start_as_current_observation(as_type="span", name="resume", metadata={"session_id": session_id}) as root:
                with propagate_attributes(session_id=session_id):
                    return await self.graph.ainvoke(Command(resume=value), config=config)
        finally:
            flush_langfuse()
