"""Retrieval agent: explicit routing, hybrid search, and evidence-gated answering.

Flow (high level):
    orchestrator → [ask_user if clarify] → [query_parser → retrieval if fetch] →
    information_evaluator → [retry orchestrator or] → answer → END

State is checkpointed per ``thread_id`` (API ``session_id``). ``messages`` holds user turns and
node ``AIMessage`` JSON (including a short retrieval status line, not passage text). Compact
passages live in ``retrieved_documents``. The information evaluator updates
``information_evaluation`` without duplicating it in ``messages``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command, interrupt

from config.settings import settings
from middleware.context_editing import truncate_and_summarize
from middleware.llm_client import get_llm_client
from observability.langfuse_handler import get_langfuse_callbacks, get_observe
from output_validation.final_answer import FinalAnswer
from output_validation.information_evaluator import InformationEvaluation
from output_validation.orchestrator import OrchestratorOutput
from output_validation.query_expansion import QueryExpansionResult
from output_validation.query_parser import QueryParserOutput
from output_validation.query_splitter import QuerySplitResult
from prompts.final_answer import SYSTEM_PROMPT as FINAL_ANSWER_PROMPT
from prompts.information_evaluator import SYSTEM_PROMPT as INFORMATION_EVALUATOR_PROMPT
from prompts.orchestrator import SYSTEM_PROMPT as ORCHESTRATOR_PROMPT
from prompts.query_expansion import SYSTEM_PROMPT as QUERY_EXPANSION_PROMPT
from prompts.query_splitter import SYSTEM_PROMPT as QUERY_SPLITTER_PROMPT
from retriever.retriever import Retriever
from tool_wrappers.prompt_plain import messages_to_plain_context
from tool_wrappers.retrieval_payload import compact_hotqa_documents_for_llm

observe = get_observe()


class RetrievalState(TypedDict, total=False):
    """Checkpointed conversation and scratch fields for one thread.

    ``messages``: user turns and structured ``AIMessage`` JSON (orchestrator, query parser,
        clarification, retrieval status notice, final answer). Passage text is not stored here;
        it lives in ``retrieved_documents``.
    ``message_summary``: rolling summary of evicted turns when context is truncated.
    ``orchestrator_output``: last ``OrchestratorOutput`` dict (query, routing flags).
    ``parsed_queries``: strings sent to the retriever after decomposition/expansion.
    ``retrieved_documents``: compact rows ``{score, text}``; extended on each retrieval hop;
        cleared on each new user ``/run`` input.
    ``information_evaluation``: last ``InformationEvaluation`` dict.
    ``information_retry_count``: increments on each incomplete evaluation (caps retries).
    """

    messages: Annotated[list, add_messages]
    message_summary: str
    orchestrator_output: dict[str, Any]
    parsed_queries: list[str]
    retrieved_documents: list[dict[str, Any]]
    information_evaluation: dict[str, Any]
    information_retry_count: int


def build_node_ai_message(
    *,
    node_name: str,
    payload: dict[str, Any],
    raw: AIMessage | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Persist structured node output as JSON on the message while keeping provider IDs/metadata."""

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
        # Shared Qdrant-backed retriever for all thread invocations on this app instance.
        self.retriever = Retriever(settings)
        self.graph = self.build_graph()

    @observe(name="orchestrator_node")
    async def orchestrator_node(self, state: RetrievalState) -> dict[str, Any]:
        """Plan routing: rewrite query, flags for clarify / retrieve / split / expand, and next edge."""

        # 1) If the thread is too long, summarize older turns and emit RemoveMessage ops.
        messages = state.get("messages", [])
        summary = state.get("message_summary", "")
        summary, kept_messages, remove_ops = await truncate_and_summarize(messages, summary)

        # 2) Prompt = rolling summary + recent transcript + last evaluator JSON (no retry counter).
        context = (
            "## Conversation Summary\n"
            f"{summary or '(none)'}\n\n"
            "## Recent Messages\n"
            f"{messages_to_plain_context(kept_messages)}\n\n"
            "## Latest Information Evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}"
        )
        # 3) Structured call: outputs OrchestratorOutput (query + booleans for downstream nodes).
        llm = get_llm_client(
            model=settings.orchestrator_model,
            output_schema=OrchestratorOutput,
            include_raw=True,
        )
        result = await llm.ainvoke(
            [
                SystemMessage(content=ORCHESTRATOR_PROMPT),
                HumanMessage(content=context),
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        response = result["parsed"]
        raw = result["raw"]
        output = response.model_dump()

        # 4) Append orchestrator JSON; clear parsed_queries so query_parser starts fresh this pass.
        return {
            "messages": remove_ops
            + [build_node_ai_message(node_name="orchestrator_node", payload=output, raw=raw)],
            "message_summary": summary,
            "orchestrator_output": output,
            "parsed_queries": [],
        }

    @observe(name="ask_user_node")
    async def ask_user_node(self, state: RetrievalState) -> dict[str, Any]:
        """Block until the client calls ``/resume`` with text; then log Q&A into the thread."""

        output = state.get("orchestrator_output") or {}
        question = str(output.get("clarification_question") or "Please clarify your request.")
        # interrupt(): API returns ``interrupted`` + ``question``; resume supplies ``answer``.
        answer = interrupt({"question": question})

        question_payload = {"clarification_question": question}
        return {
            "messages": [
                build_node_ai_message(node_name="ask_user_node", payload=question_payload),
                HumanMessage(content=f"Clarification answer: {answer}"),
            ],
        }

    @observe(name="query_parser_node")
    async def query_parser_node(self, state: RetrievalState) -> dict[str, Any]:
        """Turn orchestrator ``query`` into one or more retrieval strings (split then enrich)."""

        output = state.get("orchestrator_output") or {}
        base_query = str(output.get("query") or "").strip()
        queries = [base_query] if base_query else []
        decomposition_raw: AIMessage | None = None
        expansion_metadata: list[dict[str, Any]] = []

        # Optional multihop / multi-fact split: human message is only ``base_query`` (system prompt guides).
        if output.get("query_decomposition") and base_query:
            llm = get_llm_client(
                model=settings.query_decomposition_model,
                output_schema=QuerySplitResult,
                include_raw=True,
            )
            result = await llm.ainvoke(
                [
                    SystemMessage(content=QUERY_SPLITTER_PROMPT),
                    HumanMessage(content=base_query),
                ],
                config={"callbacks": get_langfuse_callbacks()},
            )
            split_response = result["parsed"]
            decomposition_raw = result["raw"]
            split_queries = [query.strip() for query in split_response.queries if query.strip()]
            if split_queries:
                queries = split_queries

        # Optional vocabulary bridge per sub-query: human message is that string only.
        if output.get("query_expansion") and queries:
            expanded_queries: list[str] = []
            for query in queries:
                llm = get_llm_client(
                    model=settings.query_expansion_model,
                    output_schema=QueryExpansionResult,
                    include_raw=True,
                )
                result = await llm.ainvoke(
                    [
                        SystemMessage(content=QUERY_EXPANSION_PROMPT),
                        HumanMessage(content=query),
                    ],
                    config={"callbacks": get_langfuse_callbacks()},
                )
                expansion_response = result["parsed"]
                expansion_raw = result["raw"]
                expanded_query = expansion_response.expanded_query.strip()
                if expanded_query:
                    expanded_queries.append(expanded_query)
                expansion_metadata.append(
                    {
                        "query": query,
                        "added_context": expansion_response.added_context,
                        "response_metadata": dict(
                            getattr(expansion_raw, "response_metadata", {}) or {}
                        ),
                    }
                )
            if expanded_queries:
                queries = expanded_queries

        # Persist parser summary on messages and push final list to state for ``retrieval_node``.
        parser_output = QueryParserOutput(
            queries=queries,
            decomposition_applied=bool(output.get("query_decomposition")),
            expansion_applied=bool(output.get("query_expansion")),
        ).model_dump()

        return {
            "messages": [
                build_node_ai_message(
                    node_name="query_parser_node",
                    payload=parser_output,
                    raw=decomposition_raw,
                    extra_metadata={"expansion_metadata": expansion_metadata},
                )
            ],
            "parsed_queries": parser_output["queries"],
        }

    @observe(name="retrieval_node")
    async def retrieval_node(self, state: RetrievalState) -> dict[str, Any]:
        """Hybrid Qdrant search; append compact rows to ``retrieved_documents`` plus a status line in ``messages``."""

        # Prefer query_parser output; if empty, fall back to the orchestrator’s single rewritten query.
        search_queries = [q.strip() for q in state.get("parsed_queries", []) if q.strip()]
        if not search_queries:
            fallback = str((state.get("orchestrator_output") or {}).get("query") or "").strip()
            search_queries = [fallback] if fallback else []

        ranked_hits = await self.retriever.retrieve(search_queries)
        # Same shape as ``retrieval_payload``: list of {"score", "text"} for state and evaluators.
        compact_document_rows = compact_hotqa_documents_for_llm(ranked_hits)
        accumulated = list(state.get("retrieved_documents") or [])
        updated_documents = accumulated + compact_document_rows

        retrieval_notice = {
            "detail": (
                "Retrieved documents were updated in state and are available for "
                "information completeness evaluation."
            ),
            "search_queries": search_queries,
        }
        return {
            "messages": [
                build_node_ai_message(
                    node_name="retrieval_node",
                    payload=retrieval_notice,
                )
            ],
            "retrieved_documents": updated_documents,
        }

    @observe(name="information_evaluator_node")
    async def information_evaluator_node(self, state: RetrievalState) -> dict[str, Any]:
        """Decide sufficiency from ``parsed_queries`` + accumulated ``retrieved_documents``."""

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
        retry_count = state.get("information_retry_count", 0)
        if not evaluation["information_complete"]:
            # Counts toward ``information_evaluation_max_retries`` before answer is forced.
            retry_count += 1

        # Evaluation lives only in ``information_evaluation`` (orchestrator prompt already injects it).
        return {
            "information_evaluation": evaluation,
            "information_retry_count": retry_count,
        }

    @observe(name="answer_node")
    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Emit final ``FinalAnswer`` JSON (grounded; API reads this node from ``messages``)."""

        context = (
            "## Conversation Summary\n"
            f"{state.get('message_summary') or '(none)'}\n\n"
            "## Parsed queries\n"
            f"{json.dumps(state.get('parsed_queries') or [], ensure_ascii=False)}\n\n"
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

    def route_after_orchestrator(self, state: RetrievalState) -> str:
        """Map orchestrator booleans to the next graph node name."""

        output = state.get("orchestrator_output") or {}
        if output.get("clarification_required"):
            return "ask_user"
        if output.get("retrieval_required"):
            return "query_parser"
        return "information_evaluator"

    def route_after_evaluator(self, state: RetrievalState) -> str:
        """After evaluation: answer if sufficient or retries exhausted; else replan at orchestrator."""

        evaluation = state.get("information_evaluation") or {}
        if evaluation.get("information_complete"):
            return "answer"
        if state.get("information_retry_count", 0) >= settings.information_evaluation_max_retries:
            return "answer"
        return "orchestrator"

    def build_graph(self) -> Any:
        # Nodes: orchestrator (entry) → clarify OR (query_parser→retrieval) OR direct eval →
        # evaluator → retry orchestrator or terminal answer.
        builder = StateGraph(RetrievalState)
        builder.add_node("orchestrator", self.orchestrator_node)
        builder.add_node("ask_user", self.ask_user_node)
        builder.add_node("query_parser", self.query_parser_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("information_evaluator", self.information_evaluator_node)
        builder.add_node("answer", self.answer_node)

        builder.set_entry_point("orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            self.route_after_orchestrator,
            {
                "ask_user": "ask_user",
                "query_parser": "query_parser",
                "information_evaluator": "information_evaluator",
            },
        )
        # After clarification, replan from scratch with the new HumanMessage in thread.
        builder.add_edge("ask_user", "orchestrator")
        builder.add_edge("query_parser", "retrieval")
        builder.add_edge("retrieval", "information_evaluator")
        builder.add_conditional_edges(
            "information_evaluator",
            self.route_after_evaluator,
            {
                "orchestrator": "orchestrator",
                "answer": "answer",
            },
        )
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

        # Per-turn scratch: new HumanMessage plus empty routing/evaluation/doc slots for this user message.
        return await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_query)],
                "orchestrator_output": {},
                "parsed_queries": [],
                "retrieved_documents": [],
                "information_evaluation": {},
                "information_retry_count": 0,
            },
            config=config,
        )

    async def resume(
        self,
        session_id: str,
        value: Any,
    ) -> dict[str, Any]:
        """Feed the clarification string into the paused ``interrupt()`` (same ``session_id``)."""

        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }
        callbacks = get_langfuse_callbacks()
        if callbacks:
            config["callbacks"] = callbacks

        # Value is the user’s clarification text; it becomes ``answer`` inside ``ask_user_node``.
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
