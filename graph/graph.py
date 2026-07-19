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
        → [create_queries_for_unsupported_facts → strategy_upgrade → retrieval]*
        → answer | partial_answer
        → faithfulness
        → post_deployment_metrics
        → END

Repair loop (``*``): runs while facts fail recall and ``retrieval_retry_count <
RETRIEVAL_LOOP_MAX_RETRIES``. Each repair pass uses new gap-fill queries and Qdrant
HasId exclusion via ``document_catalog`` keys — not widened top-k.

After answer/partial: one faithfulness gate checks cited ids then LLM grounding
(retry up to ``ANSWER_RETRY_MAX``). ``error_answer`` skips faithfulness and goes to
``post_deployment_metrics`` then END. Metrics node only logs a Langfuse turn snapshot.

Checkpointing
-----------
- ``messages`` and ``message_summary`` persist on the thread between turns.
- Turn scratch (``facts``, ``document_catalog``, etc.) is wiped via
  :func:`prepare_state_for_next_question` at each ``/run`` / ``/run/stream`` invoke start.

Unified fact record (in ``state["facts"]``)::

    {
        "fact_id": 1,
        "fact": "Whether X has Y",           # stable after decomposition
        "verification_status": false,        # set by recall_check
        "evidence_document_ids": [],         # catalog point ids when supported
        "search_queries": [],                # set by create_queries_for_unsupported_facts
        "gap_fill_explanation": "",
    }

Document catalog (in ``state["document_catalog"]`` during a turn)::

    {"<qdrant_point_id>": {"text": "<raw_text>", "source": "<source or empty>", "score": float}}

Module layout
-------------
- :class:`RetrievalState` — TypedDict for checkpointed state keys and reducers.
- Module-level helpers — pure fact/recall utilities (no I/O except async verify helpers).
- :class:`RetrievalGraph` — node implementations, routing, compile, ``run`` / ``stream_run``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, AsyncIterator, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# LangGraph: StateGraph builder, END sentinel, and add_messages reducer for messages key.
from langgraph.errors import NodeError
from langgraph.graph import END, StateGraph, add_messages
# Command resumes interrupted graphs.
from langgraph.types import Command, RetryPolicy

from config.settings import settings
# Optional long-context compaction (summarize + RemoveMessage); disabled below in normalisation node.
# from middleware.context_editing import truncate_and_summarize
from langfuse import propagate_attributes

from middleware.llm_client import get_llm_client
from observability.langfuse_handler import (
    flush_langfuse,
    get_langfuse_client,
    langfuse_root_observation,
    messages_for_langfuse,
    update_llm_generation,
)
# Pydantic structured-output schemas — one per LLM step in the graph.
from output_validation.fact_decomposition import RequiredFactsResult
from output_validation.faithfulness import FaithfulnessResult
from output_validation.final_answer import FinalAnswer
from output_validation.gap_fill import GapFillResult
from output_validation.recall_check import RecallVerifyResult
from output_validation.query_normalisation import QueryNormalisationResult
from output_validation.query_splitter import QuerySplitResult
from output_validation.retrieval_strategy import RetrievalStrategy
# System prompts (imported as *_PROMPT constants to avoid shadowing node method names).
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.faithfulness import SYSTEM_PROMPT as FAITHFULNESS_PROMPT
from prompts.gap_fill import SYSTEM_PROMPT as GAP_FILL_PROMPT
from prompts.fact_decomposition import SYSTEM_PROMPT as FACT_DECOMPOSITION_PROMPT
from prompts.recall_check import VERIFY_ALL_FACTS_PROMPT
from prompts.partial_answer import SYSTEM_PROMPT as PARTIAL_ANSWER_PROMPT
from prompts.query_normalisation import SYSTEM_PROMPT as QUERY_NORMALISATION_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import catalog_entries_from_retriever_hits


class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and per-turn scratch for one LangGraph thread.

    Keys marked with reducers behave specially on merge:
    - ``messages``: ``add_messages`` — append HumanMessage / AIMessage; supports RemoveMessage.

    All other keys are replaced by the latest node output.
    """

    # --- Conversation (persists across turns on the thread) ---

    messages: Annotated[list, add_messages]
    # Full chat checkpoint: user HumanMessages + final/partial AIMessage JSON only.
    # Intermediate node outputs are NOT stored here (keeps checkpoint small).

    message_summary: str
    # Rolling summary of evicted turns when ``middleware.context_editing`` truncation is
    # enabled. Currently unused (truncation disabled in query_normalisation_node).

    # --- Turn scratch (reset at each /run invoke via prepare_state_for_next_question) ---

    user_question: str
    # Raw user utterance for this turn (tracking / Langfuse only; not used by prompts).

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

    document_catalog: dict[str, dict[str, Any]]
    # Turn-local map point_id → {text, source, score}; merged across retrieval passes.

    strategies_used: list[str]
    # Retrieval strategies attempted this turn (seeded at query_complexity, appended on upgrade).

    # --- Fact scratch (reset each /run; recall_check mutates verification fields) ---

    facts: list[dict[str, Any]]
    # Unified fact records from fact_decomposition; verification updated by recall_check.

    needs_split: bool
    # True when len(facts) > 1; set by query_complexity for routing and SSE/UI.

    recall_sufficient: bool
    # True when every fact has verification_status=True after recall_check.
    # Drives route_after_recall_check (answer vs repair vs partial_answer).

    # --- Repair loop budget ---

    retrieval_retry_count: int
    # Incremented by strategy_upgrade each repair pass. Compared against
    # settings.retrieval_loop_max_retries to gate partial_answer.

    graph_failure: dict[str, Any]
    # Node-level retry exhaustion context from ``handle_node_failure``.

    # --- Answer / faithfulness scratch ---

    cited_document_ids: list[str]
    # Point ids returned by answer/partial_answer for grounding.

    answer_text: str
    # Latest answer/partial text; faithfulness reads this (no message scraping).

    answer_mode: str
    # ``full`` or ``partial`` — faithfulness retries return to the same mode.

    faithfulness_retry_count: int
    # Increments on each faithfulness failure (id check or LLM grounding).

    faithfulness_ok: bool
    # True when the answer may end (pass or forced pass after retry exhaustion).

    faithfulness_forced_pass: bool
    # True when faithfulness shipped after ANSWER_RETRY_MAX exhaustion (not a clean pass).

    faithfulness_feedback: str
    # Retry hint for answer/partial; empty string means omit from the LLM prompt.

    final_sources: list[str]
    # Code-built source labels after faithfulness (non-empty catalog.source only).


ERROR_ANSWER_USER_MESSAGE = "An error occurred while processing your request. Please try again."


def handle_node_failure(state: RetrievalState, error: NodeError) -> Command:
    """Route retry-exhausted node failures to deterministic ``error_answer`` output."""

    return Command(
        update={"graph_failure": {"failed_node": error.node, "detail": str(error.error)}},
        goto="error_answer",
    )


# --- Fact and recall helpers ---
#
# Pure functions (except verify_all_facts which calls the LLM).
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


def ensure_evidence_document_ids_in_catalog(
    facts: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]] | None,
) -> None:
    """Raise if any fact's ``evidence_document_ids`` are missing from ``document_catalog``.

    Used after recall_check LLM merge so invalid ids fail the node and trigger RetryPolicy.
    """

    catalog = catalog or {}
    keys = set(catalog.keys())
    invalid: list[str] = []
    seen: set[str] = set()
    for row in facts:
        for point_id in row.get("evidence_document_ids") or []:
            key = str(point_id).strip()
            if key and key in keys:
                continue
            label = str(point_id)
            if label in seen:
                continue
            seen.add(label)
            invalid.append(label)
    if invalid:
        raise ValueError(
            "recall_check evidence_document_ids not in document_catalog: "
            f"{json.dumps(invalid, ensure_ascii=False)}"
        )


async def verify_all_facts(
    facts: list[dict[str, Any]],
    *,
    normalized_query: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Verify every fact in one batched recall-check LLM call.

    Uses ``RecallVerifyResult`` and merges verdicts onto input rows by ``fact_id``.
    Raises if any ``evidence_document_ids`` are missing from ``catalog`` (node retry).
    """

    if not facts:
        raise ValueError("verify_all_facts requires at least one fact")

    verifier = get_llm_client(
        model=settings.recall_check_model,
        output_schema=RecallVerifyResult,
        include_raw=True,
    )
    facts_for_prompt = [
        {"fact_id": row.get("fact_id"), "fact": row.get("fact")} for row in facts
    ]
    human_content = (
        f"## Normalized query\n{normalized_query}\n\n"
        f"## Facts to verify\n"
        f"{json.dumps(facts_for_prompt, ensure_ascii=False)}\n\n"
        f"## Document catalog\n{catalog_docs_block(catalog)}"
    )
    messages_for_llm = [
        SystemMessage(content=VERIFY_ALL_FACTS_PROMPT),
        HumanMessage(content=human_content),
    ]
    llm_response = await verifier.ainvoke(messages_for_llm)
    verdicts = llm_response["parsed"].facts

    verdict_by_id = {
        verdict.fact_id: verdict for verdict in verdicts if verdict.fact_id is not None
    }

    updated: list[dict[str, Any]] = []
    for row in facts:
        fact_id = row.get("fact_id")
        verdict = verdict_by_id.get(fact_id)
        if verdict is None:
            raise ValueError(
                f"recall_check batch output missing verification for fact_id={fact_id!r}"
            )
        updated.append(
            {
                **row,
                "verification_status": verdict.verification_status,
                "evidence_document_ids": list(verdict.evidence_document_ids),
            }
        )
    ensure_evidence_document_ids_in_catalog(updated, catalog)
    return updated


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
            "evidence_document_ids": [],
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


def merge_document_catalog(
    existing: dict[str, dict[str, Any]] | None,
    new_rows: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Merge catalog maps; existing keys win on id collision."""

    merged = dict(existing or {})
    for point_id, row in (new_rows or {}).items():
        key = str(point_id).strip()
        if not key or key in merged:
            continue
        merged[key] = row
    return merged


def catalog_texts_for_prompt(catalog: dict[str, dict[str, Any]] | None) -> list[str]:
    """Return non-empty catalog passage texts for gap-fill prompts."""

    texts: list[str] = []
    for row in (catalog or {}).values():
        text = str((row or {}).get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def catalog_docs_block(catalog: dict[str, dict[str, Any]] | None) -> str:
    """Numbered passage block from document_catalog for recall/answer prompts."""

    lines: list[str] = []
    for index, (point_id, row) in enumerate((catalog or {}).items(), start=1):
        row = row or {}
        text = str(row.get("text") or "")
        score = row.get("score")
        lines.append(f"[{index}] (id={point_id}, score={score}) {text}")
    return "\n".join(lines) if lines else "(none)"


def build_sources_from_catalog(
    catalog: dict[str, dict[str, Any]] | None,
    cited_ids: list[str] | None,
) -> list[str]:
    """Collect non-empty source labels for cited ids (skip missing / empty source)."""

    catalog = catalog or {}
    sources: list[str] = []
    seen: set[str] = set()
    for point_id in cited_ids or []:
        key = str(point_id).strip()
        row = catalog.get(key) or {}
        source = str(row.get("source") or "").strip()
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def cited_ids_valid(
    cited_ids: list[str] | None,
    catalog: dict[str, dict[str, Any]] | None,
) -> tuple[bool, list[str]]:
    """Return (ok, invalid_ids) for cited_document_ids against catalog keys."""

    keys = set((catalog or {}).keys())
    invalid: list[str] = []
    for point_id in cited_ids or []:
        key = str(point_id).strip()
        if not key or key not in keys:
            invalid.append(str(point_id))
    return (len(invalid) == 0, invalid)


def prepare_state_for_next_question(user_query: str) -> dict[str, Any]:
    """Build the state patch for a new user turn before ``graph.ainvoke``.

    Appends one ``HumanMessage`` and wipes all per-turn scratch so a prior turn's facts,
    catalog, and retry counters do not carry over on the same thread.

    Args:
        user_query: Raw user message text for this turn.

    Returns:
        State merge dict with ``messages`` plus empty scratch fields.
    """

    return {
        "messages": [HumanMessage(content=user_query)],
        "user_question": user_query,
        "normalized_query": "",
        "retrieval_strategy": "",
        "active_retrieval_queries": [],
        "document_catalog": {},
        "strategies_used": [],
        "facts": [],
        "needs_split": False,
        "recall_sufficient": False,
        "retrieval_retry_count": 0,
        "graph_failure": {},
        "cited_document_ids": [],
        "answer_text": "",
        "answer_mode": "",
        "faithfulness_retry_count": 0,
        "faithfulness_ok": False,
        "faithfulness_forced_pass": False,
        "faithfulness_feedback": "",
        "final_sources": [],
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
            f"{messages_to_plain_context(kept_messages)}"
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
                node_span.update(input={"user_question": state.get("user_question") or "", "messages": messages_for_langfuse(state.get("messages")), "message_summary": state.get("message_summary") or ""}, output={"normalized_query": normalized_query})
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
                node_span.update(input={"normalized_query": normalized_query}, output={"facts": facts})
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
        active_retrieval_queries = [normalized_query] if normalized_query else []

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="query_complexity") as node_span:
                node_span.update(input={"facts": facts}, output={"needs_split": needs_split, "fact_count": fact_count, "active_retrieval_queries": active_retrieval_queries})

        return {
            "needs_split": needs_split,
            "explanation": explanation,
            "fact_count": fact_count,
            "retrieval_strategy": "fast_bm25_retrieval",
            "strategies_used": ["fast_bm25_retrieval"],
            "active_retrieval_queries": active_retrieval_queries,
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
                node_span.update(input={"normalized_query": normalized_query, "facts": facts}, output={"active_retrieval_queries": queries})
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            queries = [query.strip() for query in response.queries if query.strip()]
            if not queries and normalized_query:
                queries = [normalized_query]

        return {"active_retrieval_queries": queries}

    # --- Retrieval, recall check, and repair loop ---

    async def retrieval_node(self, state: RetrievalState) -> dict[str, Any]:
        """Retrieve passages from Qdrant; merge into document_catalog.

        Purpose:
            Fetch candidate chunks for recall_check and answer generation. During repair
            loops, new queries exclude already-seen point ids (HasId must_not) from
            ``document_catalog`` keys so each pass adds fresh evidence.

        Reads:
            ``active_retrieval_queries``, ``retrieval_strategy``, ``normalized_query``,
            ``document_catalog``

        Writes:
            ``document_catalog`` — merged map of new point ids.
            ``retrieval_strategy``, ``active_retrieval_queries``

        Routes to:
            ``recall_check`` (fixed edge).
        """

        raw_strategy = str(state.get("retrieval_strategy") or "").strip()
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

        existing_catalog = dict(state.get("document_catalog") or {})
        exclude_ids = list(existing_catalog.keys()) or None

        async def run_retrieval() -> dict[str, dict[str, Any]]:
            ranked_hits = await self.retriever.retrieve(
                search_queries,
                strategy=strategy,
                top_k=settings.retrieval_top_k,
                dense_mmr_limit=settings.retrieval_candidate_dense_mmr,
                bm25_limit=settings.retrieval_candidate_bm25,
                late_interaction_limit=settings.retrieval_candidate_for_late_interaction,
                exclude_point_ids=exclude_ids,
            )
            new_entries = catalog_entries_from_retriever_hits(ranked_hits)
            return merge_document_catalog(existing_catalog, new_entries)

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="retrieval") as node_span:
                merged_catalog = await run_retrieval()
                node_span.update(input={"active_retrieval_queries": search_queries, "retrieval_strategy": strategy}, output={"document_catalog": merged_catalog})
        else:
            merged_catalog = await run_retrieval()

        return {
            "document_catalog": merged_catalog,
            "retrieval_strategy": strategy,
            "active_retrieval_queries": search_queries,
        }

    # --- Recall / repair nodes ---

    async def recall_check_node(self, state: RetrievalState) -> dict[str, Any]:
        """Verify all facts in one batched LLM call against retrieved passages; update in place.

        Input (from ``state``):
            normalized_query: str — standalone query for this turn.
            facts: list[dict] — unified fact rows from fact_decomposition, e.g.::

                {
                    "fact_id": 1,
                    "fact": "Whether X has Y",
                    "verification_status": false,
                    "evidence_document_ids": [],
                    "search_queries": [],
                    "gap_fill_explanation": "",
                }

            document_catalog: map of point id → {text, source, score}.

        Output (state merge):
            facts: list[dict] — same rows with verification fields set per fact.
            recall_sufficient: bool — ``True`` only when every fact has
                ``verification_status`` true.

        Raises when evidence_document_ids are not in document_catalog (node RetryPolicy).

        Routes via: ``route_after_recall_check`` (answer | partial_answer | create_queries_for_unsupported_facts).
        """

        normalized_query = str(state.get("normalized_query") or "").strip()
        facts = list(state.get("facts") or [])
        if not facts:
            raise ValueError("recall_check requires facts from fact_decomposition")

        catalog = dict(state.get("document_catalog") or {})

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="recall_check") as node_span:
                verified_facts = await verify_all_facts(
                    facts,
                    normalized_query=normalized_query,
                    catalog=catalog,
                )
                recall_sufficient = all(row["verification_status"] for row in verified_facts)
                node_span.update(input={"facts": facts, "document_catalog": catalog}, output={"facts": verified_facts})
        else:
            verified_facts = await verify_all_facts(
                facts,
                normalized_query=normalized_query,
                catalog=catalog,
            )
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
            ``document_catalog`` (prior corpus texts only in the prompt).

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
            "## Document catalog texts\n"
            f"{json.dumps(catalog_texts_for_prompt(state.get('document_catalog') or {}), ensure_ascii=False)}"
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
                node_span.update(input={"unsupported_facts": unsupported, "document_catalog": state.get("document_catalog") or {}}, output={"facts": updated_facts, "active_retrieval_queries": queries})
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

        prior_strategy = str(state.get("retrieval_strategy") or "")
        prior_retry_count = int(state.get("retrieval_retry_count") or 0)
        retry_count = prior_retry_count + 1
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
                node_span.update(input={"retrieval_strategy": prior_strategy, "retrieval_retry_count": prior_retry_count}, output={"retrieval_strategy": strategy, "retrieval_retry_count": retry_count})

        return {
            "retrieval_strategy": strategy,
            "retrieval_retry_count": retry_count,
            "strategies_used": list(state.get("strategies_used") or []) + [strategy],
        }

    # --- Answer and cleanup ---

    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Synthesize a grounded final answer when all facts passed recall verification.

        Writes ``answer_mode=full``, ``cited_document_ids``, and an AIMessage JSON payload.
        Routes to ``faithfulness``.
        """

        catalog = dict(state.get("document_catalog") or {})
        feedback = str(state.get("faithfulness_feedback") or "").strip()
        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Document catalog\n"
            f"{json.dumps(catalog, ensure_ascii=False)}"
        )
        if feedback:
            context = f"## Faithfulness feedback\n{feedback}\n\n{context}"
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
                answer["sources"] = []
                node_span.update(input={"user_question": state.get("user_question") or "", "normalized_query": state.get("normalized_query") or "", "facts": state.get("facts") or [], "document_catalog": catalog}, output={"answer": answer.get("answer") or "", "cited_document_ids": answer.get("cited_document_ids") or []})
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            raw = result["raw"]
            answer = response.model_dump()
            answer["sources"] = []
        return {
            "answer_mode": "full",
            "answer_text": str(answer.get("answer") or ""),
            "cited_document_ids": list(answer.get("cited_document_ids") or []),
            "faithfulness_feedback": "",
            "messages": [
                build_node_ai_message(node_name="answer_node", payload=answer, raw=raw)
            ],
        }

    async def partial_answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit grounded partial answer when retrieval retry budgets are exhausted.

        Writes ``answer_mode=partial`` and routes to ``faithfulness``.
        """

        catalog = dict(state.get("document_catalog") or {})
        feedback = str(state.get("faithfulness_feedback") or "").strip()
        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Facts\n"
            f"{json.dumps(state.get('facts') or [], ensure_ascii=False)}\n\n"
            "## Document catalog\n"
            f"{json.dumps(catalog, ensure_ascii=False)}"
        )
        if feedback:
            context = f"## Faithfulness feedback\n{feedback}\n\n{context}"
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
                answer["sources"] = []
                node_span.update(input={"user_question": state.get("user_question") or "", "normalized_query": state.get("normalized_query") or "", "facts": state.get("facts") or [], "document_catalog": catalog}, output={"answer": answer.get("answer") or "", "cited_document_ids": answer.get("cited_document_ids") or []})
        else:
            result = await llm.ainvoke(messages_for_llm)
            response = result["parsed"]
            raw = result["raw"]
            answer = response.model_dump()
            answer["sources"] = []
        return {
            "answer_mode": "partial",
            "answer_text": str(answer.get("answer") or ""),
            "cited_document_ids": list(answer.get("cited_document_ids") or []),
            "faithfulness_feedback": "",
            "messages": [
                build_node_ai_message(node_name="partial_answer_node", payload=answer, raw=raw)
            ],
        }

    async def error_answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit deterministic user-facing fallback after node retries are exhausted.

        Skips faithfulness (routes straight to END).
        """

        payload = FinalAnswer(
            answer=ERROR_ANSWER_USER_MESSAGE,
            cited_document_ids=[],
            sources=[],
            confidence="low",
        ).model_dump()
        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="error_answer") as node_span:
                node_span.update(input={"graph_failure": state.get("graph_failure") or {}}, output=payload)
        return {
            "cited_document_ids": [],
            "final_sources": [],
            "messages": [build_node_ai_message(node_name="error_answer_node", payload=payload)],
        }

    async def faithfulness_node(self, state: RetrievalState) -> dict[str, Any]:
        """Gate the answer before END: cited ids must exist, then LLM grounding.

        Purpose:
            After ``answer`` / ``partial_answer``, decide whether the turn can finish.
            Code checks cited ids against ``document_catalog``; if valid, an LLM checks
            that ``answer_text`` is supported by those cited passages alone.
            On failure, set feedback and retry the same answer mode until
            ``ANSWER_RETRY_MAX``, then force-pass with sources from valid cited ids.
            API reads sources from ``final_sources`` (message JSON keeps ``sources: []``).

        Input (state):
            answer_text, cited_document_ids, document_catalog, faithfulness_retry_count,
            answer_mode (used by ``route_after_faithfulness``).

        Output (state merge):
            faithfulness_ok, faithfulness_feedback, faithfulness_retry_count,
            final_sources (on pass / force-pass).

        Routes via: ``route_after_faithfulness`` (post_deployment_metrics | answer | partial_answer).
        """

        catalog = dict(state.get("document_catalog") or {})
        cited_ids = [
            str(point_id).strip()
            for point_id in (state.get("cited_document_ids") or [])
            if str(point_id).strip()
        ]
        answer_text = str(state.get("answer_text") or "")

        # --- Step 1: code gate — every cited id must be a catalog key (no LLM) ---
        ids_ok, invalid_ids = cited_ids_valid(cited_ids, catalog)
        if not ids_ok:
            feedback = (
                "Cited document id validation failed. These ids are not in document_catalog: "
                f"{json.dumps(invalid_ids, ensure_ascii=False)}. "
                "Regenerate the answer using only valid catalog point ids. sources must stay []."
            )
            retry_count = int(state.get("faithfulness_retry_count") or 0) + 1
            if retry_count >= settings.answer_retry_max:
                # Budget exhausted: ship last answer; sources only from ids that exist.
                valid_cited = [point_id for point_id in cited_ids if point_id in catalog]
                merge: dict[str, Any] = {
                    "faithfulness_ok": True,
                    "faithfulness_forced_pass": True,
                    "faithfulness_feedback": "",
                    "faithfulness_retry_count": retry_count,
                    "final_sources": build_sources_from_catalog(catalog, valid_cited),
                }
            else:
                # Retry: answer/partial sees faithfulness_feedback in its human context.
                merge = {
                    "faithfulness_ok": False,
                    "faithfulness_feedback": feedback,
                    "faithfulness_retry_count": retry_count,
                }
            if settings.langfuse_tracing_enabled:
                langfuse = get_langfuse_client()
                with langfuse.start_as_current_observation(as_type="span", name="faithfulness") as node_span:
                    node_span.update(
                        input={"answer": answer_text, "cited_document_ids": cited_ids, "document_catalog": catalog},
                        output={"faithfulness_ok": merge.get("faithfulness_ok"), "invalid_ids": invalid_ids, "faithfulness_retry_count": merge.get("faithfulness_retry_count"), "final_sources": merge.get("final_sources")},
                    )
            return merge

        # --- Step 2: LLM gate — is answer_text grounded in the cited passages only? ---
        cited_passages = {
            point_id: str((catalog.get(point_id) or {}).get("text") or "")
            for point_id in cited_ids
        }
        human_content = (
            f"## Answer\n{answer_text}\n\n"
            f"## Cited passages\n{json.dumps(cited_passages, ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.faithfulness_model,
            output_schema=FaithfulnessResult,
            include_raw=True,
        )
        messages_for_llm = [
            SystemMessage(content=FAITHFULNESS_PROMPT),
            HumanMessage(content=human_content),
        ]
        model = settings.faithfulness_model

        if settings.langfuse_tracing_enabled:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(as_type="span", name="faithfulness") as node_span:
                with langfuse.start_as_current_observation(as_type="generation", name="faithfulness-llm", model=model) as gen:
                    result = await llm.ainvoke(messages_for_llm)
                    update_llm_generation(gen, model=model, raw=result.get("raw"))
                parsed = result["parsed"]
                if parsed.passed:
                    # Pass: fill final_sources from cited catalog rows; turn can END.
                    merge = {
                        "faithfulness_ok": True,
                        "faithfulness_feedback": "",
                        "final_sources": build_sources_from_catalog(catalog, cited_ids),
                    }
                else:
                    feedback = (
                        "Faithfulness check failed. The answer is not fully supported by the cited "
                        f"passages. Reason: {parsed.reason.strip()}. "
                        "Regenerate a grounded answer. cited_document_ids must be valid catalog ids; "
                        "sources must stay []."
                    )
                    retry_count = int(state.get("faithfulness_retry_count") or 0) + 1
                    if retry_count >= settings.answer_retry_max:
                        # Force-pass after grounding failures exhaust the retry budget.
                        valid_cited = [point_id for point_id in cited_ids if point_id in catalog]
                        merge = {
                            "faithfulness_ok": True,
                            "faithfulness_forced_pass": True,
                            "faithfulness_feedback": "",
                            "faithfulness_retry_count": retry_count,
                            "final_sources": build_sources_from_catalog(catalog, valid_cited),
                        }
                    else:
                        merge = {
                            "faithfulness_ok": False,
                            "faithfulness_feedback": feedback,
                            "faithfulness_retry_count": retry_count,
                        }
                node_span.update(
                    input={"answer": answer_text, "cited_passages": cited_passages},
                    output={"faithfulness_ok": merge.get("faithfulness_ok"), "reason": parsed.reason, "faithfulness_retry_count": merge.get("faithfulness_retry_count"), "final_sources": merge.get("final_sources")},
                )
                return merge

        # Same LLM path without Langfuse spans.
        result = await llm.ainvoke(messages_for_llm)
        parsed = result["parsed"]
        if parsed.passed:
            return {
                "faithfulness_ok": True,
                "faithfulness_feedback": "",
                "final_sources": build_sources_from_catalog(catalog, cited_ids),
            }

        feedback = (
            "Faithfulness check failed. The answer is not fully supported by the cited "
            f"passages. Reason: {parsed.reason.strip()}. "
            "Regenerate a grounded answer. cited_document_ids must be valid catalog ids; "
            "sources must stay []."
        )
        retry_count = int(state.get("faithfulness_retry_count") or 0) + 1
        if retry_count >= settings.answer_retry_max:
            valid_cited = [point_id for point_id in cited_ids if point_id in catalog]
            return {
                "faithfulness_ok": True,
                "faithfulness_forced_pass": True,
                "faithfulness_feedback": "",
                "faithfulness_retry_count": retry_count,
                "final_sources": build_sources_from_catalog(catalog, valid_cited),
            }
        return {
            "faithfulness_ok": False,
            "faithfulness_feedback": feedback,
            "faithfulness_retry_count": retry_count,
        }

    async def post_deployment_metrics_node(self, state: RetrievalState) -> dict[str, Any]:
        """Log end-of-turn eval fields as a Langfuse child span under run/stream_run.

        Also mirrors the snapshot onto the root span output when available.
        No LLM. No answer mutation. When Langfuse is off, this is a no-op.
        Hidden from SSE/UI progress.
        """

        if not settings.langfuse_tracing_enabled:
            return {}

        confidence = "low"
        for message in reversed(state.get("messages") or []):
            name = getattr(message, "name", None)
            if name not in {"answer_node", "partial_answer_node", "error_answer_node"}:
                continue
            content = getattr(message, "content", None)
            try:
                payload = json.loads(content) if isinstance(content, str) else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("confidence") is not None:
                confidence = str(payload.get("confidence") or "low")
            break

        facts = list(state.get("facts") or [])
        unsupported_fact_ids = [
            row.get("fact_id") for row in facts if not row.get("verification_status")
        ]
        snapshot = {
            "user_question": state.get("user_question") or "",
            "normalized_query": state.get("normalized_query") or "",
            "document_catalog": dict(state.get("document_catalog") or {}),
            "strategies_used": list(state.get("strategies_used") or []),
            "retrieval_retry_count": int(state.get("retrieval_retry_count") or 0),
            "faithfulness_retry_count": int(state.get("faithfulness_retry_count") or 0),
            "fact_count": len(facts),
            "fact_ids": [row.get("fact_id") for row in facts],
            "unsupported_fact_ids": unsupported_fact_ids,
            "recall_sufficient": bool(state.get("recall_sufficient")),
            "answer_mode": str(state.get("answer_mode") or ""),
            "answer_text": str(state.get("answer_text") or ""),
            "cited_document_ids": list(state.get("cited_document_ids") or []),
            "final_sources": list(state.get("final_sources") or []),
            "confidence": confidence,
            "faithfulness_ok": bool(state.get("faithfulness_ok")),
            "faithfulness_forced_pass": bool(state.get("faithfulness_forced_pass")),
            "graph_failure": dict(state.get("graph_failure") or {}),
        }

        langfuse = get_langfuse_client()
        with langfuse.start_as_current_observation(as_type="span", name="post_deployment_metrics") as node_span:
            node_span.update(output=snapshot)

        root = langfuse_root_observation.get()
        if root is not None:
            root.update(output=snapshot)
        return {}

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

    def route_after_faithfulness(self, state: RetrievalState) -> str:
        """Metrics then END when faithfulness_ok; otherwise retry the same answer_mode."""

        if state.get("faithfulness_ok"):
            return "post_deployment_metrics"
        if str(state.get("answer_mode") or "") == "partial":
            return "partial_answer"
        return "answer"

    # --- Graph wiring ---

    def build_graph(self) -> Any:
        """Wire nodes and compile the LangGraph with a session checkpointer.

        Repair loop: recall_check → create_queries_for_unsupported_facts → strategy_upgrade → retrieval → recall_check.
        Answer path: answer|partial_answer → faithfulness → post_deployment_metrics → END (or retry answer mode).
        Retry exhaustion on configured nodes: handle_node_failure → error_answer → post_deployment_metrics → END.
        """
        builder = StateGraph(RetrievalState)
        node_retry = RetryPolicy(max_attempts=settings.graph_node_retry_max_attempts, initial_interval=1.0, backoff_factor=2.0)
        builder.add_node("query_normalisation", self.query_normalisation_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("fact_decomposition", self.fact_decomposition_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("query_complexity", self.query_complexity_node)
        builder.add_node("query_splitter", self.query_splitter_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("retrieval", self.retrieval_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("recall_check", self.recall_check_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("create_queries_for_unsupported_facts", self.create_queries_for_unsupported_facts_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("strategy_upgrade", self.strategy_upgrade_node)
        builder.add_node("answer", self.answer_node, retry_policy=node_retry)
        builder.add_node("partial_answer", self.partial_answer_node, retry_policy=node_retry)
        builder.add_node("faithfulness", self.faithfulness_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("error_answer", self.error_answer_node)
        builder.add_node("post_deployment_metrics", self.post_deployment_metrics_node)

        builder.set_entry_point("query_normalisation")
        builder.add_edge("query_normalisation", "fact_decomposition")
        builder.add_edge("fact_decomposition", "query_complexity")
        builder.add_conditional_edges(
            "query_complexity",
            self.route_after_complexity,
            {
                "retrieval": "retrieval",
                "query_splitter": "query_splitter",
            },
        )
        builder.add_edge("query_splitter", "retrieval")
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
        builder.add_edge("answer", "faithfulness")
        builder.add_edge("partial_answer", "faithfulness")
        builder.add_conditional_edges(
            "faithfulness",
            self.route_after_faithfulness,
            {
                "answer": "answer",
                "partial_answer": "partial_answer",
                "post_deployment_metrics": "post_deployment_metrics",
            },
        )
        builder.add_edge("error_answer", "post_deployment_metrics")
        builder.add_edge("post_deployment_metrics", END)
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
                root_token = langfuse_root_observation.set(root)
                try:
                    with propagate_attributes(session_id=session_id):
                        return await self.graph.ainvoke(invoke_input, config=config)
                finally:
                    langfuse_root_observation.reset(root_token)
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
                root_token = langfuse_root_observation.set(root)
                try:
                    with propagate_attributes(session_id=session_id):
                        async for update in self.graph.astream(
                            invoke_input,
                            config=config,
                            stream_mode="updates",
                        ):
                            yield update
                finally:
                    langfuse_root_observation.reset(root_token)
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
                root_token = langfuse_root_observation.set(root)
                try:
                    with propagate_attributes(session_id=session_id):
                        return await self.graph.ainvoke(Command(resume=value), config=config)
                finally:
                    langfuse_root_observation.reset(root_token)
        finally:
            flush_langfuse()
