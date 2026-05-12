from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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
    """LangGraph state for retrieval conversations."""

    messages: Annotated[list, add_messages]
    message_summary: str
    orchestrator_output: dict[str, Any]
    parsed_queries: list[str]
    information_evaluation: dict[str, Any]
    information_retry_count: int


def build_node_ai_message(
    *,
    node_name: str,
    payload: dict[str, Any],
    raw: AIMessage | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Store validated node output in messages while preserving raw provider metadata."""

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
    """Wrapper around the explicit LangGraph retrieval pipeline."""

    def __init__(self, checkpointer: Any) -> None:
        self.checkpointer = checkpointer
        self._postgres_context: Any | None = None
        self.retriever = Retriever(settings)
        self.graph = self.build_graph()

    @classmethod
    async def create(cls) -> "RetrievalGraph":
        """Create the graph with either in-memory or Postgres checkpointing."""

        if settings.checkpointer_use_postgres:
            if not settings.database_url.strip():
                raise ValueError(
                    "CHECKPOINTER_USE_POSTGRES=true but DATABASE_URL is missing."
                )

            postgres_context = AsyncPostgresSaver.from_conn_string(settings.database_url)
            checkpointer = await postgres_context.__aenter__()
            await checkpointer.setup()
            instance = cls(checkpointer)
            instance._postgres_context = postgres_context
            print("RAG checkpointer: Postgres", flush=True)
            return instance

        print("RAG checkpointer: InMemorySaver", flush=True)
        return cls(InMemorySaver())

    @observe(name="orchestrator_node")
    async def orchestrator_node(self, state: RetrievalState) -> dict[str, Any]:
        """Plan the next route and produce one retrieval-ready query when needed."""

        messages = state.get("messages", [])
        summary = state.get("message_summary", "")
        summary, kept_messages, remove_ops = await truncate_and_summarize(messages, summary)

        context = (
            "## Conversation Summary\n"
            f"{summary or '(none)'}\n\n"
            "## Recent Messages\n"
            f"{messages_to_plain_context(kept_messages)}\n\n"
            "## Latest Information Evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
            "## Information Retry Count\n"
            f"{state.get('information_retry_count', 0)} of {settings.information_evaluation_max_retries}"
        )
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

        return {
            "messages": remove_ops
            + [build_node_ai_message(node_name="orchestrator_node", payload=output, raw=raw)],
            "message_summary": summary,
            "orchestrator_output": output,
            "parsed_queries": [],
        }

    @observe(name="ask_user_node")
    async def ask_user_node(self, state: RetrievalState) -> dict[str, Any]:
        """Pause for a human clarification and append the clarification exchange on resume."""

        output = state.get("orchestrator_output") or {}
        question = str(output.get("clarification_question") or "Please clarify your request.")
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
        """Apply decomposition first, then expansion, and overwrite parsed_queries."""

        output = state.get("orchestrator_output") or {}
        base_query = str(output.get("query") or "").strip()
        queries = [base_query] if base_query else []
        decomposition_raw: AIMessage | None = None
        expansion_metadata: list[dict[str, Any]] = []

        if output.get("query_decomposition") and base_query:
            context = (
                "## Conversation Summary\n"
                f"{state.get('message_summary') or '(none)'}\n\n"
                "## Messages\n"
                f"{messages_to_plain_context(state.get('messages', []))}\n\n"
                "## Latest Orchestrator Output\n"
                f"{json.dumps(state.get('orchestrator_output') or {}, ensure_ascii=False)}\n\n"
                "## Latest Information Evaluation\n"
                f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
                "## Information Retry Count\n"
                f"{state.get('information_retry_count', 0)} of {settings.information_evaluation_max_retries}\n\n"
                "## Query To Decompose\n"
                f"{base_query}"
            )
            llm = get_llm_client(
                model=settings.query_decomposition_model,
                output_schema=QuerySplitResult,
                include_raw=True,
            )
            result = await llm.ainvoke(
                [
                    SystemMessage(content=QUERY_SPLITTER_PROMPT),
                    HumanMessage(content=context),
                ],
                config={"callbacks": get_langfuse_callbacks()},
            )
            split_response = result["parsed"]
            decomposition_raw = result["raw"]
            split_queries = [query.strip() for query in split_response.queries if query.strip()]
            if split_queries:
                queries = split_queries

        if output.get("query_expansion") and queries:
            expanded_queries: list[str] = []
            for query in queries:
                context = (
                    "## Conversation Summary\n"
                    f"{state.get('message_summary') or '(none)'}\n\n"
                    "## Messages\n"
                    f"{messages_to_plain_context(state.get('messages', []))}\n\n"
                    "## Latest Orchestrator Output\n"
                    f"{json.dumps(state.get('orchestrator_output') or {}, ensure_ascii=False)}\n\n"
                    "## Latest Information Evaluation\n"
                    f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
                    "## Information Retry Count\n"
                    f"{state.get('information_retry_count', 0)} of {settings.information_evaluation_max_retries}\n\n"
                    "## Query To Expand\n"
                    f"{query}"
                )
                llm = get_llm_client(
                    model=settings.query_expansion_model,
                    output_schema=QueryExpansionResult,
                    include_raw=True,
                )
                result = await llm.ainvoke(
                    [
                        SystemMessage(content=QUERY_EXPANSION_PROMPT),
                        HumanMessage(content=context),
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
        """Retrieve documents for all parsed queries and store results as a ToolMessage."""

        queries = [query.strip() for query in state.get("parsed_queries", []) if query.strip()]
        if not queries:
            query = str((state.get("orchestrator_output") or {}).get("query") or "").strip()
            queries = [query] if query else []

        docs = await self.retriever.retrieve(queries)
        payload = {
            "queries": queries,
            "documents": compact_hotqa_documents_for_llm(docs),
        }
        return {
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    name="retrieval_node",
                    tool_call_id=f"retrieval-{uuid4()}",
                    response_metadata={"node": "retrieval_node"},
                )
            ],
        }

    @observe(name="information_evaluator_node")
    async def information_evaluator_node(self, state: RetrievalState) -> dict[str, Any]:
        """Check whether the accumulated messages contain enough evidence to answer."""

        context = (
            "## Conversation Summary\n"
            f"{state.get('message_summary') or '(none)'}\n\n"
            "## Messages\n"
            f"{messages_to_plain_context(state.get('messages', []))}\n\n"
            "## Latest Orchestrator Output\n"
            f"{json.dumps(state.get('orchestrator_output') or {}, ensure_ascii=False)}\n\n"
            "## Latest Information Evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
            "## Information Retry Count\n"
            f"{state.get('information_retry_count', 0)} of {settings.information_evaluation_max_retries}"
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
        raw = result["raw"]
        evaluation = response.model_dump()
        retry_count = state.get("information_retry_count", 0)
        if not evaluation["information_complete"]:
            retry_count += 1

        return {
            "messages": [
                build_node_ai_message(
                    node_name="information_evaluator_node",
                    payload=evaluation,
                    raw=raw,
                )
            ],
            "information_evaluation": evaluation,
            "information_retry_count": retry_count,
        }

    @observe(name="answer_node")
    async def answer_node(self, state: RetrievalState) -> dict[str, Any]:
        """Produce the final API-compatible answer JSON from the message audit trail."""

        context = (
            "## Conversation Summary\n"
            f"{state.get('message_summary') or '(none)'}\n\n"
            "## Messages\n"
            f"{messages_to_plain_context(state.get('messages', []))}\n\n"
            "## Latest Orchestrator Output\n"
            f"{json.dumps(state.get('orchestrator_output') or {}, ensure_ascii=False)}\n\n"
            "## Latest Information Evaluation\n"
            f"{json.dumps(state.get('information_evaluation') or {}, ensure_ascii=False)}\n\n"
            "## Information Retry Count\n"
            f"{state.get('information_retry_count', 0)} of {settings.information_evaluation_max_retries}"
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
        """Select the next node from the orchestrator's structured output."""

        output = state.get("orchestrator_output") or {}
        if output.get("clarification_required"):
            return "ask_user"
        if output.get("retrieval_required"):
            return "query_parser"
        return "information_evaluator"

    def route_after_evaluator(self, state: RetrievalState) -> str:
        """Retry retrieval until information is complete or the configured limit is hit."""

        evaluation = state.get("information_evaluation") or {}
        if evaluation.get("information_complete"):
            return "answer"
        if state.get("information_retry_count", 0) >= settings.information_evaluation_max_retries:
            return "answer"
        return "orchestrator"

    def build_graph(self) -> Any:
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
        """Invoke one retrieval-agent turn and reset per-question routing state."""

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
                "orchestrator_output": {},
                "parsed_queries": [],
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
        """Resume after ask_user interrupted the graph."""

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
        return self.graph.get_state({"configurable": {"thread_id": session_id}})

    async def close(self) -> None:
        await self.retriever.qdrant.close()
        if self._postgres_context is not None:
            await self._postgres_context.__aexit__(None, None, None)
