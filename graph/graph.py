"""Retrieval agent: normalize, classify, retrieve, evaluate, and answer.

Flow (high level):
    query_normalisation -> query_complexity -> retrieval preparation -> retrieval ->
    information_evaluator -> [gap fill / intent correction retry or] -> answer -> END

State is checkpointed per ``thread_id`` (API ``session_id``). ``messages`` stays lean:
it stores user turns and final answer messages only. Intermediate node outputs live in
node-specific state keys such as ``normalized_query``, ``parsed_queries``,
``retrieved_documents``, and ``information_evaluation``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command

from config.settings import settings
from middleware.context_editing import truncate_and_summarize
from middleware.llm_client import get_llm_client
from observability.langfuse_handler import get_langfuse_callbacks, get_observe
from output_validation.final_answer import FinalAnswer
from output_validation.gap_fill import GapFillResult
from output_validation.information_evaluator import InformationEvaluation
from output_validation.intent_correction_rewriter import IntentCorrectionRewriteResult
from output_validation.query_complexity import QueryComplexityResult
from output_validation.query_expansion import QueryExpansionResult
from output_validation.query_normalisation import QueryNormalisationResult
from output_validation.query_rewriter import QueryRewriteResult
from output_validation.query_splitter import QuerySplitResult
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.gap_fill import SYSTEM_PROMPT as GAP_FILL_PROMPT
from prompts.information_evaluator import SYSTEM_PROMPT as INFORMATION_EVALUATOR_PROMPT
from prompts.intent_correction_rewriter import (
    SYSTEM_PROMPT as INTENT_CORRECTION_REWRITER_PROMPT,
)
from prompts.query_complexity import SYSTEM_PROMPT as QUERY_COMPLEXITY_PROMPT
from prompts.query_expansion import SYSTEM_PROMPT as QUERY_EXPANSION_PROMPT
from prompts.query_normalisation import SYSTEM_PROMPT as QUERY_NORMALISATION_PROMPT
from prompts.query_rewriter import SYSTEM_PROMPT as QUERY_REWRITER_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import compact_hotqa_documents_for_llm

observe = get_observe()


class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and scratch fields for one thread.

    ``messages``: user turns and final answer ``AIMessage`` JSON. Intermediate graph outputs are
        intentionally not appended here.
    ``message_summary``: rolling summary of evicted turns when context is truncated.
    ``normalized_query``: latest user query rewritten into standalone form.
    ``query_complexity``: last ``QueryComplexityResult`` dict used for routing.
    ``parsed_queries``: primary retrieval queries produced by simple routing, splitting,
        expansion, ambiguous rewriting, or intent correction.
    ``parsed_queries_insufficient_recall``: gap-fill retrieval queries generated after an
        insufficient recall evaluation.
    ``retrieval_query_source``: state key the retrieval node should read for the next pass.
    ``retrieved_documents``: compact rows ``{score, text}``, appended across retry loops and
        reset for each new user ``/run`` input.
    ``information_evaluation``: last ``InformationEvaluation`` dict.
    ``missing_evidence_details``: evaluator gap details for insufficient recall.
    ``insufficient_recall_retry_count``: count of recall-repair loops in the current turn.
    ``intent_mismatch_retry_count``: count of intent-correction loops in the current turn.
    """

    messages: Annotated[list, add_messages]
    message_summary: str
    normalized_query: str
    query_complexity: dict[str, Any]
    parsed_queries: list[str]
    parsed_queries_insufficient_recall: list[str]
    retrieval_query_source: str
    retrieved_documents: list[dict[str, Any]]
    information_evaluation: dict[str, Any]
    missing_evidence_details: list[str]
    insufficient_recall_retry_count: int
    intent_mismatch_retry_count: int


def build_node_ai_message(
    *,
    node_name: str,
    payload: dict[str, Any],
    raw: AIMessage | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Persist final structured output as JSON while keeping provider IDs/metadata."""

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
        self.checkpointer = checkpointer
        self._postgres_context = postgres_context
        self.retriever = Retriever(settings)
        self.graph = self.build_graph()

    @observe(name="query_normalisation_node")
    async def query_normalisation_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite the latest user message into a standalone query with conversation context."""

        messages = state.get("messages", [])
        summary = state.get("message_summary", "")
        summary, kept_messages, remove_ops = await truncate_and_summarize(messages, summary)

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

        return {
            "messages": remove_ops,
            "message_summary": summary,
            "normalized_query": normalized_query,
        }

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

        return {
            "query_complexity": output,
            "parsed_queries": [normalized_query] if normalized_query else [],
            "parsed_queries_insufficient_recall": [],
            "retrieval_query_source": "parsed_queries",
        }

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

        return {
            "parsed_queries": queries,
            "parsed_queries_insufficient_recall": [],
            "retrieval_query_source": "parsed_queries",
        }

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

        return {
            "parsed_queries": queries,
            "parsed_queries_insufficient_recall": [],
            "retrieval_query_source": "parsed_queries",
        }

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

        return {
            "parsed_queries": [rewritten_query] if rewritten_query else [],
            "parsed_queries_insufficient_recall": [],
            "retrieval_query_source": "parsed_queries",
        }

    @observe(name="retrieval_node")
    async def retrieval_node(self, state: RetrievalState) -> dict[str, Any]:
        """Retrieve for the active parsed query key and append compact docs to state."""

        source = state.get("retrieval_query_source") or "parsed_queries"
        if source == "parsed_queries_insufficient_recall":
            search_queries = [
                query.strip()
                for query in state.get("parsed_queries_insufficient_recall", [])
                if query.strip()
            ]
        else:
            search_queries = [
                query.strip()
                for query in state.get("parsed_queries", [])
                if query.strip()
            ]

        if not search_queries:
            fallback = str(state.get("normalized_query") or "").strip()
            search_queries = [fallback] if fallback else []

        ranked_hits = await self.retriever.retrieve(search_queries)
        compact_document_rows = compact_hotqa_documents_for_llm(ranked_hits)
        accumulated = list(state.get("retrieved_documents") or [])

        return {
            "retrieved_documents": accumulated + compact_document_rows,
        }

    @observe(name="information_evaluator_node")
    async def information_evaluator_node(self, state: RetrievalState) -> dict[str, Any]:
        """Classify retrieval as sufficient, insufficient recall, or intent mismatch."""

        context = (
            "## Parsed queries\n"
            f"{json.dumps(state.get('parsed_queries') or [], ensure_ascii=False)}\n\n"
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
        response = result["parsed"]
        evaluation = response.model_dump()
        status = evaluation["evaluation_status"]

        insufficient_recall_retry_count = state.get("insufficient_recall_retry_count", 0)
        intent_mismatch_retry_count = state.get("intent_mismatch_retry_count", 0)
        if status == "insufficient_recall":
            insufficient_recall_retry_count += 1
        elif status == "intent_mismatch":
            intent_mismatch_retry_count += 1

        missing_evidence_details = (
            evaluation["missing_evidence_details"]
            if status == "insufficient_recall"
            else []
        )

        return {
            "information_evaluation": evaluation,
            "missing_evidence_details": missing_evidence_details,
            "insufficient_recall_retry_count": insufficient_recall_retry_count,
            "intent_mismatch_retry_count": intent_mismatch_retry_count,
        }

    @observe(name="gap_fill_node")
    async def gap_fill_node(self, state: RetrievalState) -> dict[str, Any]:
        """Generate targeted missing-evidence queries after insufficient recall."""

        context = (
            "## Parsed queries\n"
            f"{json.dumps(state.get('parsed_queries') or [], ensure_ascii=False)}\n\n"
            "## Retrieved documents\n"
            f"{json.dumps(state.get('retrieved_documents') or [], ensure_ascii=False)}\n\n"
            "## Missing evidence details\n"
            f"{json.dumps(state.get('missing_evidence_details') or [], ensure_ascii=False)}"
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
        if not queries:
            queries = [query for query in state.get("parsed_queries", []) if query.strip()]

        return {
            "parsed_queries_insufficient_recall": queries,
            "retrieval_query_source": "parsed_queries_insufficient_recall",
        }

    @observe(name="intent_correction_rewriter_node")
    async def intent_correction_rewriter_node(self, state: RetrievalState) -> dict[str, Any]:
        """Rewrite retrieval queries when the evaluator detects intent mismatch."""

        normalized_query = str(state.get("normalized_query") or "").strip()
        context = (
            "## Normalized query\n"
            f"{normalized_query}\n\n"
            "## Parsed queries\n"
            f"{json.dumps(state.get('parsed_queries') or [], ensure_ascii=False)}\n\n"
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
        if not queries:
            queries = [query for query in state.get("parsed_queries", []) if query.strip()]
        if not queries and normalized_query:
            queries = [normalized_query]

        return {
            "parsed_queries": queries,
            "parsed_queries_insufficient_recall": [],
            "retrieval_query_source": "parsed_queries",
        }

    @observe(name="answer_node")
    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit final ``FinalAnswer`` JSON; API reads this node from ``messages``."""

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
        return {
            "messages": [
                build_node_ai_message(node_name="answer_node", payload=answer, raw=raw)
            ]
        }

    def route_after_complexity(self, state: RetrievalState) -> str:
        """Map exact complexity labels to the next graph node name."""

        complexity = (state.get("query_complexity") or {}).get("complexity")
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
                return "answer"
            return "gap_fill"
        if status == "intent_mismatch":
            if (
                state.get("intent_mismatch_retry_count", 0)
                >= settings.intent_mismatch_max_retries
            ):
                return "answer"
            return "intent_correction_rewriter"
        return "answer"

    def build_graph(self) -> Any:
        # Nodes: normalize -> classify -> prepare retrieval queries -> retrieval ->
        # evaluator -> retry repair or terminal answer.
        builder = StateGraph(RetrievalState)
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

        builder.set_entry_point("query_normalisation")
        builder.add_edge("query_normalisation", "query_complexity")
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
        builder.add_conditional_edges(
            "information_evaluator",
            self.route_after_evaluator,
            {
                "answer": "answer",
                "gap_fill": "gap_fill",
                "intent_correction_rewriter": "intent_correction_rewriter",
            },
        )
        builder.add_edge("gap_fill", "retrieval")
        builder.add_edge("intent_correction_rewriter", "retrieval")
        builder.add_edge("answer", END)
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

        return await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_query)],
                "normalized_query": "",
                "query_complexity": {},
                "parsed_queries": [],
                "parsed_queries_insufficient_recall": [],
                "retrieval_query_source": "parsed_queries",
                "retrieved_documents": [],
                "information_evaluation": {},
                "missing_evidence_details": [],
                "insufficient_recall_retry_count": 0,
                "intent_mismatch_retry_count": 0,
            },
            config=config,
        )

    async def resume(
        self,
        session_id: str,
        value: Any,
    ) -> dict[str, Any]:
        """Feed a future clarification string into a paused graph interrupt."""

        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }
        callbacks = get_langfuse_callbacks()
        if callbacks:
            config["callbacks"] = callbacks

        return await self.graph.ainvoke(
            Command(resume=value),
            config=config,
        )

    def get_state(self, session_id: str) -> Any:
        """Inspect checkpointed graph state for debugging or tooling (optional)."""

        return self.graph.get_state({"configurable": {"thread_id": session_id}})

    async def close(self) -> None:
        # Qdrant client + Postgres saver context must be closed on process shutdown.
        await self.retriever.qdrant.close()
        if self._postgres_context is not None:
            await self._postgres_context.__aexit__(None, None, None)
