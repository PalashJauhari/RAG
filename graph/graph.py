"""Retrieval agent: normalize, classify, retrieve, evaluate, and answer.

Flow (high level):
    query_normalisation -> query_complexity -> retrieval preparation -> retrieval ->
    information_evaluator -> [gap fill / intent correction / strategy upgrade retry or] ->
    answer/partial_answer -> clear_turn_trace -> END

State is checkpointed per ``thread_id`` (API ``session_id``). ``messages`` stays lean:
it stores user turns and final/partial answer messages only. Intermediate node outputs live in
node-specific state keys such as ``normalized_query``, ``active_retrieval_queries``,
``retrieval_strategy``, ``message_query``, ``retrieved_documents``, and ``information_evaluation``.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, AsyncIterator, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command, Overwrite

from config.settings import settings
# Optional long-context compaction (summarize + RemoveMessage); disabled below in normalisation node.
# from middleware.context_editing import truncate_and_summarize
from middleware.llm_client import get_llm_client
from observability.langfuse_handler import get_langfuse_callbacks, get_observe
from output_validation.final_answer import FinalAnswer
from output_validation.gap_fill import GapFillResult
from output_validation.information_evaluator import (
    InformationEvaluation,
    resolve_strategy_upgrade,
)
from output_validation.intent_correction_rewriter import IntentCorrectionRewriteResult
from output_validation.message_query_entry import MessageQueryEntry
from output_validation.query_complexity import QueryComplexityResult
from output_validation.query_expansion import QueryExpansionResult
from output_validation.query_normalisation import QueryNormalisationResult
from output_validation.query_rewriter import QueryRewriteResult
from output_validation.query_splitter import QuerySplitResult
from output_validation.retrieval_strategy import RetrievalStrategy
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.gap_fill import SYSTEM_PROMPT as GAP_FILL_PROMPT
from prompts.information_evaluator import SYSTEM_PROMPT as INFORMATION_EVALUATOR_PROMPT
from prompts.intent_correction_rewriter import (
    SYSTEM_PROMPT as INTENT_CORRECTION_REWRITER_PROMPT,
)
from prompts.partial_answer import SYSTEM_PROMPT as PARTIAL_ANSWER_PROMPT
from prompts.query_complexity import SYSTEM_PROMPT as QUERY_COMPLEXITY_PROMPT
from prompts.query_expansion import SYSTEM_PROMPT as QUERY_EXPANSION_PROMPT
from prompts.query_normalisation import SYSTEM_PROMPT as QUERY_NORMALISATION_PROMPT
from prompts.query_rewriter import SYSTEM_PROMPT as QUERY_REWRITER_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import compact_hotqa_documents_for_llm

# Langfuse spans when tracing is on; no-op decorator otherwise.
observe = get_observe()


class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and scratch fields for one thread.

    ``messages``: user turns and final answer ``AIMessage`` JSON. Intermediate graph outputs are
        intentionally not appended here.
    ``message_summary``: rolling summary of evicted turns when context is truncated.
    ``normalized_query``: latest user query rewritten into standalone form.
    ``query_complexity``: last ``QueryComplexityResult`` dict used for routing (includes retrieval_strategy).
    ``retrieval_strategy``: scalar tier applied by ``retrieval_node`` (may change after evaluator/gap/intent).
    ``active_retrieval_queries``: current retrieval query strings for this turn.
    ``message_query``: append-only structured audit trace (``operator.add`` reducer).
    ``retrieved_documents``: compact rows ``{score, text}``, appended across retry loops and
        reset for each new user ``/run`` input.
    ``new_retrieved_documents``: compact rows from the latest retrieval pass only (for SSE / UI).
    ``information_evaluation``: last ``InformationEvaluation`` dict (includes missing_evidence_details when applicable).
    ``insufficient_recall_retry_count``: count of recall-repair loops in the current turn.
    ``intent_mismatch_retry_count``: count of intent-correction loops in the current turn.
    ``strategy_upgrade_retry_count``: count of evaluator-driven retrieval tier upgrades this turn.
    """

    # LangGraph merges updates; add_messages appends new msgs and applies RemoveMessage ops.
    messages: Annotated[list, add_messages]
    message_summary: str
    normalized_query: str
    query_complexity: dict[str, Any]
    retrieval_strategy: str
    active_retrieval_queries: list[str]
    message_query: Annotated[list[dict[str, Any]], operator.add]
    retrieved_documents: list[dict[str, Any]]
    new_retrieved_documents: list[dict[str, Any]]
    information_evaluation: dict[str, Any]
    insufficient_recall_retry_count: int
    intent_mismatch_retry_count: int
    strategy_upgrade_retry_count: int


def trace_row(node: str, kind: str, payload: dict[str, Any], notes: str | None = None) -> dict[str, Any]:
    """Return a single ``message_query`` append fragment."""

    entry = MessageQueryEntry(node=node, kind=kind, payload=payload, notes=notes)
    return {"message_query": [entry.model_dump()]}


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


class RetrievalGraph:
    """Compiles the pipeline with a graph checkpointer (memory or Postgres).

    Pass a ready checkpointer from app startup (see ``api.main`` lifespan for Postgres vs memory).
    """

    def __init__(self, checkpointer: Any, postgres_context: Any | None = None) -> None:
        # postgres_context is only set when using AsyncPostgresSaver so close() can exit the conn ctx.
        self.checkpointer = checkpointer
        self._postgres_context = postgres_context
        self.retriever = Retriever(settings)
        self.graph = self.build_graph()

    @observe(name="query_normalisation_node")
    async def query_normalisation_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite the latest user message into a standalone query with conversation context."""

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
        result = await llm.ainvoke(
            [
                SystemMessage(content=QUERY_NORMALISATION_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        output = response.model_dump()
        normalized_query = output["normalized_query"].strip() or latest_user_query.strip()

        # remove_ops shrink checkpoint when truncation is on; message_summary holds rolled-up history.
        merge = {
            "messages": remove_ops,
            "message_summary": summary,
            "normalized_query": normalized_query,
        }
        merge.update(trace_row("query_normalisation", "normalisation", {}))
        return merge

    @observe(name="query_complexity_node")
    async def query_complexity_node(self, state: RetrievalState) -> dict[str, Any]:
        """Classify normalized query complexity and seed simple-query parsed queries."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        llm = get_llm_client(
            model=settings.query_complexity_model,
            output_schema=QueryComplexityResult,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=QUERY_COMPLEXITY_PROMPT),
                HumanMessage(content=normalized_query),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        output = response.model_dump()
        retrieval_strategy = output["retrieval_strategy"]

        # Seed retrieval state for the simple path (complexity → retrieval with no splitter/expander).
        merge = {
            "query_complexity": output,
            "retrieval_strategy": retrieval_strategy,
            "active_retrieval_queries": [normalized_query] if normalized_query else [],
        }
        merge.update(
            trace_row(
                "query_complexity",
                "complexity",
                {"complexity": output["complexity"]},
                notes=output.get("explanation"),
            )
        )
        return merge

    @observe(name="query_splitter_node")
    async def query_splitter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Split comparison, multihop, and procedural queries into focused retrieval strings."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        llm = get_llm_client(
            model=settings.query_decomposition_model,
            output_schema=QuerySplitResult,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=QUERY_SPLITTER_PROMPT),
                HumanMessage(content=normalized_query),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        queries = [query.strip() for query in response.queries if query.strip()]
        if not queries and normalized_query:
            queries = [normalized_query]

        merge = {"active_retrieval_queries": queries}
        merge.update(trace_row("query_splitter", "query_prep", {"queries": queries}))
        return merge

    @observe(name="query_expansion_node")
    async def query_expansion_node(self, state: RetrievalState) -> dict[str, Any]:
        """Create multiple retrieval angles for exploratory queries."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        llm = get_llm_client(
            model=settings.query_expansion_model,
            output_schema=QueryExpansionResult,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=QUERY_EXPANSION_PROMPT),
                HumanMessage(content=normalized_query),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        queries = [query.strip() for query in response.queries if query.strip()]
        if not queries and normalized_query:
            queries = [normalized_query]

        merge = {"active_retrieval_queries": queries}
        merge.update(trace_row("query_expansion", "query_prep", {"queries": queries}))
        return merge

    @observe(name="query_rewriter_node")
    async def query_rewriter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite ambiguous queries without interrupting for clarification yet."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        llm = get_llm_client(
            model=settings.query_rewriter_model,
            output_schema=QueryRewriteResult,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=QUERY_REWRITER_PROMPT),
                HumanMessage(content=normalized_query),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        rewritten_query = response.rewritten_query.strip() or normalized_query

        queries = [rewritten_query] if rewritten_query else []
        merge = {"active_retrieval_queries": queries}
        merge.update(trace_row("query_rewriter", "query_prep", {"queries": queries}))
        return merge

    @observe(name="retrieval_node")
    async def retrieval_node(self, state: RetrievalState) -> dict[str, Any]:
        """Retrieve using ``active_retrieval_queries`` and ``retrieval_strategy``; append docs."""

        raw_strategy = str(state.get("retrieval_strategy") or "").strip()
        strategy: RetrievalStrategy = (
            raw_strategy
            if raw_strategy
            in {
                "fast_retrieval",
                "fast_bm25_retrieval",
                "keyword",
                "fast_bm25_late_interaction_retrieval",
            }
            else "fast_bm25_retrieval"
        )

        search_queries = [
            query.strip()
            for query in state.get("active_retrieval_queries") or []
            if query.strip()
        ]

        if not search_queries:
            fallback = str(state.get("normalized_query") or "").strip()
            search_queries = [fallback] if fallback else []

        ranked_hits = await self.retriever.retrieve(search_queries, strategy=strategy)
        compact_document_rows = compact_hotqa_documents_for_llm(ranked_hits)
        accumulated = list(state.get("retrieved_documents") or [])

        seen_texts = {doc.get("text") for doc in accumulated if doc.get("text") is not None}
        rows_to_add: list[dict[str, Any]] = []
        for row in compact_document_rows:
            text = row.get("text")
            if text is None or text in seen_texts:
                continue
            rows_to_add.append(row)
            seen_texts.add(text)

        merge = {
            "retrieved_documents": accumulated + rows_to_add,
            "retrieval_strategy": strategy,
            "new_retrieved_documents": compact_document_rows,
            "active_retrieval_queries": search_queries,
        }
        merge.update(
            trace_row(
                "retrieval",
                "retrieval",
                {
                    "queries": search_queries,
                    "strategy": strategy,
                    "new_doc_count": len(compact_document_rows),
                },
            )
        )
        return merge

    @observe(name="information_evaluator_node")
    async def information_evaluator_node(self, state: RetrievalState) -> dict[str, Any]:
        """Classify retrieval as sufficient, insufficient recall, or intent mismatch."""

        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Retrieval strategy\n"
            f"{state.get('retrieval_strategy') or ''}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Message query trace\n"
            f"{json.dumps(state.get('message_query') or [], ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.information_evaluator_model,
            output_schema=InformationEvaluation,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=INFORMATION_EVALUATOR_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        evaluation_model: InformationEvaluation = result["parsed"]
        force_partial = False
        if evaluation_model.evaluation_status == "strategy_upgrade":
            evaluation_model, force_partial = resolve_strategy_upgrade(
                str(state.get("retrieval_strategy") or ""),
                evaluation_model,
                max_upgrade_retries=settings.strategy_upgrade_max_retries,
            )
        evaluation = evaluation_model.model_dump()
        status = evaluation["evaluation_status"]

        # route_after_evaluator uses these counts vs settings.*_max_retries to stop retry loops.
        insufficient_recall_retry_count = state.get("insufficient_recall_retry_count", 0)
        intent_mismatch_retry_count = state.get("intent_mismatch_retry_count", 0)
        strategy_upgrade_retry_count = state.get("strategy_upgrade_retry_count", 0)
        if status == "insufficient_recall":
            insufficient_recall_retry_count += 1
        elif status == "intent_mismatch":
            intent_mismatch_retry_count += 1
        elif status == "strategy_upgrade":
            if force_partial:
                strategy_upgrade_retry_count = settings.strategy_upgrade_max_retries
            else:
                strategy_upgrade_retry_count += 1

        merge: dict[str, Any] = {
            "information_evaluation": evaluation,
            "insufficient_recall_retry_count": insufficient_recall_retry_count,
            "intent_mismatch_retry_count": intent_mismatch_retry_count,
            "strategy_upgrade_retry_count": strategy_upgrade_retry_count,
        }
        if status == "strategy_upgrade" and not force_partial:
            merge["retrieval_strategy"] = evaluation["next_retrieval_strategy"]

        merge.update(
            trace_row(
                "information_evaluator",
                "evaluation",
                {"evaluation_status": status},
                notes=evaluation.get("evaluation_explanation"),
            )
        )
        return merge

    @observe(name="gap_fill_node")
    async def gap_fill_node(self, state: RetrievalState) -> dict[str, Any]:
        """Generate targeted missing-evidence queries after insufficient recall."""

        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Retrieval strategy\n"
            f"{state.get('retrieval_strategy') or ''}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Message query trace\n"
            f"{json.dumps(state.get('message_query') or [], ensure_ascii=False)}\n\n"
            "## Information evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.gap_fill_model,
            output_schema=GapFillResult,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=GAP_FILL_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        queries = [query.strip() for query in response.missing_queries if query.strip()]
        previous_active = [q for q in state.get("active_retrieval_queries") or [] if q.strip()]
        if not queries:
            queries = previous_active

        next_rs = response.next_retrieval_strategy
        prior_tier = str(state.get("retrieval_strategy") or "")
        tier = next_rs if next_rs is not None else prior_tier
        merge: dict[str, Any] = {
            "active_retrieval_queries": queries,
            "retrieval_strategy": tier,
        }
        merge.update(
            trace_row(
                "gap_fill",
                "gap_fill",
                {"queries": queries, "next_retrieval_strategy": next_rs},
                notes=response.gap_fill_explanation,
            )
        )
        return merge

    @observe(name="intent_correction_rewriter_node")
    async def intent_correction_rewriter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite retrieval queries when the evaluator detects intent mismatch."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        context = (
            "## Normalized query\n"
            f"{normalized_query}\n\n"
            "## Retrieval strategy\n"
            f"{state.get('retrieval_strategy') or ''}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Message query trace\n"
            f"{json.dumps(state.get('message_query') or [], ensure_ascii=False)}\n\n"
            "## Information evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.intent_correction_rewriter_model,
            output_schema=IntentCorrectionRewriteResult,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=INTENT_CORRECTION_REWRITER_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        queries = [query.strip() for query in response.corrected_queries if query.strip()]
        previous_active = [q for q in state.get("active_retrieval_queries") or [] if q.strip()]
        if not queries:
            queries = previous_active
        if not queries and normalized_query:
            queries = [normalized_query]

        next_rs = response.next_retrieval_strategy
        prior_tier = str(state.get("retrieval_strategy") or "")
        tier = next_rs if next_rs is not None else prior_tier
        merge: dict[str, Any] = {
            "active_retrieval_queries": queries,
            "retrieval_strategy": tier,
        }
        merge.update(
            trace_row(
                "intent_correction_rewriter",
                "intent_correction",
                {"queries": queries, "next_retrieval_strategy": next_rs},
                notes=response.correction_explanation,
            )
        )
        return merge

    @observe(name="answer_node")
    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit final ``FinalAnswer`` JSON; API reads this node from ``messages``."""

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
        result = await llm.ainvoke(
            [
                SystemMessage(content=FINAL_ANSWER_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        raw = result["raw"]
        answer = response.model_dump()
        # API scans messages for final answer node names and JSON matching FinalAnswer (see api.main).
        return {
            "messages": [
                build_node_ai_message(node_name="answer_node", payload=answer, raw=raw)
            ]
        }

    @observe(name="partial_answer_node")
    async def partial_answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit a grounded partial answer after evaluator retry budgets are exhausted."""

        context = (
            "## Normalized query\n"
            f"{state.get('normalized_query') or ''}\n\n"
            "## Retrieval strategy\n"
            f"{state.get('retrieval_strategy') or ''}\n\n"
            "## Active retrieval queries\n"
            f"{json.dumps(state.get('active_retrieval_queries') or [], ensure_ascii=False)}\n\n"
            "## Information evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}"
        )
        llm = get_llm_client(
            model=settings.final_answer_model,
            output_schema=FinalAnswer,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=PARTIAL_ANSWER_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        raw = result["raw"]
        answer = response.model_dump()
        return {
            "messages": [
                build_node_ai_message(node_name="partial_answer_node", payload=answer, raw=raw)
            ]
        }

    @observe(name="clear_turn_trace_node")
    async def clear_turn_trace_node(self, state: RetrievalState) -> dict[str, Any]:
        """Clear append-only audit trace so the next user turn starts fresh on this thread."""

        return {"message_query": Overwrite([])}

    def route_after_complexity(self, state: RetrievalState) -> str:
        """Map exact complexity labels to the next graph node name."""

        complexity = (state.get("query_complexity") or {}).get("complexity")
        # simple_query (and anything unexpected) falls through to retrieval using seeded active_retrieval_queries.
        if complexity in {"comparison_query", "multihop_query", "procedural_query"}:
            return "query_splitter"
        if complexity == "exploratory_query":
            return "query_expansion"
        if complexity == "ambiguous_query":
            return "query_rewriter"
        return "retrieval"

    def route_after_evaluator(self, state: RetrievalState) -> str:
        """Route to answer, recall gap fill, or intent correction after evaluation."""

        evaluation = state.get("information_evaluation") or {}
        status = evaluation.get("evaluation_status")
        if status == "sufficient":
            return "answer"
        if status == "insufficient_recall":
            if (
                state.get("insufficient_recall_retry_count", 0)
                >= settings.insufficient_recall_max_retries
            ):
                return "partial_answer"
            # Loop: gap_fill sets insufficient_recall queries then retrieval runs again.
            return "gap_fill"
        if status == "intent_mismatch":
            if (
                state.get("intent_mismatch_retry_count", 0)
                >= settings.intent_mismatch_max_retries
            ):
                return "partial_answer"
            # Loop: intent_correction_rewriter sets intent-correction queries then retrieval runs again.
            return "intent_correction_rewriter"
        if status == "strategy_upgrade":
            if (
                state.get("strategy_upgrade_retry_count", 0)
                >= settings.strategy_upgrade_max_retries
            ):
                return "partial_answer"
            # Same queries; heavier retrieval_strategy merged by evaluator output before this routes.
            return "retrieval"
        # Unknown status: fail closed to answer rather than spinning retries forever.
        return "answer"

    def build_graph(self) -> Any:
        # Linear spine: normalisation → complexity → (optional prep) → retrieval → evaluator → answer.
        # Side loops: evaluator → gap_fill → retrieval; evaluator → intent_correction → retrieval.
        builder = StateGraph(RetrievalState)
        # Internal node ids match strings returned by route_after_* for conditional_edges.
        builder.add_node("query_normalisation", self.query_normalisation_node)
        builder.add_node("query_complexity", self.query_complexity_node)
        builder.add_node("query_splitter", self.query_splitter_node)
        builder.add_node("query_expansion", self.query_expansion_node)
        builder.add_node("query_rewriter", self.query_rewriter_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("information_evaluator", self.information_evaluator_node)
        builder.add_node("gap_fill", self.gap_fill_node)
        builder.add_node("intent_correction_rewriter", self.intent_correction_rewriter_node)
        builder.add_node("answer", self.answer_node)
        builder.add_node("partial_answer", self.partial_answer_node)
        builder.add_node("clear_turn_trace", self.clear_turn_trace_node)

        builder.set_entry_point("query_normalisation")
        builder.add_edge("query_normalisation", "query_complexity")
        # Dict keys must equal route_after_complexity return values (retrieval | query_splitter | ...).
        builder.add_conditional_edges(
            "query_complexity",
            self.route_after_complexity,
            {
                "retrieval": "retrieval",
                "query_splitter": "query_splitter",
                "query_expansion": "query_expansion",
                "query_rewriter": "query_rewriter",
            },
        )
        builder.add_edge("query_splitter", "retrieval")
        builder.add_edge("query_expansion", "retrieval")
        builder.add_edge("query_rewriter", "retrieval")
        builder.add_edge("retrieval", "information_evaluator")
        # Dict keys must equal route_after_evaluator return values.
        builder.add_conditional_edges(
            "information_evaluator",
            self.route_after_evaluator,
            {
                "answer": "answer",
                "partial_answer": "partial_answer",
                "gap_fill": "gap_fill",
                "intent_correction_rewriter": "intent_correction_rewriter",
                "retrieval": "retrieval",
            },
        )
        builder.add_edge("gap_fill", "retrieval")
        builder.add_edge("intent_correction_rewriter", "retrieval")
        builder.add_edge("answer", "clear_turn_trace")
        builder.add_edge("partial_answer", "clear_turn_trace")
        builder.add_edge("clear_turn_trace", END)
        # Persists checkpoints keyed by thread_id (session_id from the API).
        return builder.compile(checkpointer=self.checkpointer)

    async def run(
        self,
        session_id: str,
        user_query: str,
    ) -> dict[str, Any]:
        """Start or continue a thread: append user text; reset scratch fields for this turn."""

        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }
        callbacks = get_langfuse_callbacks()
        if callbacks:
            config["callbacks"] = callbacks

        # Turn-local scratch is wiped each invoke; messages reducer merges this HumanMessage onto the thread.
        return await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_query)],
                "normalized_query": "",
                "query_complexity": {},
                "retrieval_strategy": "",
                "active_retrieval_queries": [],
                "retrieved_documents": [],
                "new_retrieved_documents": [],
                "information_evaluation": {},
                "insufficient_recall_retry_count": 0,
                "intent_mismatch_retry_count": 0,
                "strategy_upgrade_retry_count": 0,
                "message_query": Overwrite([]),
            },
            config=config,
        )

    async def stream_run(
        self,
        session_id: str,
        user_query: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Start a thread turn and yield compact node updates as the graph progresses."""

        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }
        callbacks = get_langfuse_callbacks()
        if callbacks:
            config["callbacks"] = callbacks

        # Same input contract as run(); only the execution method changes from ainvoke to astream.
        async for update in self.graph.astream(
            {
                "messages": [HumanMessage(content=user_query)],
                "normalized_query": "",
                "query_complexity": {},
                "retrieval_strategy": "",
                "active_retrieval_queries": [],
                "retrieved_documents": [],
                "new_retrieved_documents": [],
                "information_evaluation": {},
                "insufficient_recall_retry_count": 0,
                "intent_mismatch_retry_count": 0,
                "strategy_upgrade_retry_count": 0,
                "message_query": Overwrite([]),
            },
            config=config,
            stream_mode="updates",
        ):
            yield update

    async def resume(
        self,
        session_id: str,
        value: Any,
    ) -> dict[str, Any]:
        """Feed a future clarification string into a paused graph interrupt."""

        # Resume path for graphs that call interrupt(); the shipped agent completes without pausing today.
        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }
        callbacks = get_langfuse_callbacks()
        if callbacks:
            config["callbacks"] = callbacks

        # LangGraph resumes from the saved interrupt marker for this thread_id.
        return await self.graph.ainvoke(
            Command(resume=value),
            config=config,
        )

    def get_state(self, session_id: str) -> Any:
        """Inspect checkpointed graph state for debugging or tooling (optional)."""

        # Latest checkpoint snapshot for thread_id without advancing the graph.
        return self.graph.get_state({"configurable": {"thread_id": session_id}})

    async def close(self) -> None:
        # Release HTTP resources (Qdrant) and async Postgres saver connection context if used.
        await self.retriever.qdrant.close()
        if self._postgres_context is not None:
            await self._postgres_context.__aexit__(None, None, None)
