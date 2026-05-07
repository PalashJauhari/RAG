from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from config.settings import settings
from middleware.context_editing import truncate_and_summarize
from middleware.llm_client import get_llm_client
from observability.langfuse_handler import get_langfuse_callbacks
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from prompts.orchestrator import SYSTEM_PROMPT
from tools.ask_user import ask_user
from tools.query_expansion import query_expansion
from tools.query_rewriter import query_rewriter
from tools.query_splitter import query_splitter
from tools.retrieval_tool import retrieval_tool


class RetrievalState(TypedDict, total=False):
    """LangGraph state for retrieval conversations."""

    messages: Annotated[list, add_messages]
    message_summary: str


TOOLS = [
    query_rewriter,
    query_expansion,
    query_splitter,
    retrieval_tool,
    ask_user,
]


class RetrievalGraph:
    """Wrapper around the compiled LangGraph retrieval graph."""

    def __init__(self, checkpointer: Any) -> None:
        self.checkpointer = checkpointer
        self.callbacks = get_langfuse_callbacks()
        self._postgres_context: Any | None = None
        self.llm_with_tools = get_llm_client(json_mode=True).bind_tools(TOOLS)
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

    async def orchestrator(self, state: RetrievalState) -> dict[str, Any]:
        """Prepare retrieval context, call tools when needed, and produce final JSON."""

        messages = state.get("messages", [])
        summary = state.get("message_summary", "")
        summary, kept_messages, remove_ops = await truncate_and_summarize(messages, summary)

        context = (
            "## Conversation Summary\n"
            f"{summary or '(none)'}\n\n"
            "## Recent Messages\n"
            f"{kept_messages}\n\n"
            "Use this summary only as conversation context. Retrieve documents before answering."
        )
        response = await self.llm_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]
        )

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        ask_user_calls = [call for call in tool_calls if call.get("name") == "ask_user"]
        if ask_user_calls and len(tool_calls) > 1:
            response = AIMessage(content=response.content or "", tool_calls=[ask_user_calls[0]])

        return {
            "messages": remove_ops + [response],
            "message_summary": summary,
        }

    def should_continue(self, state: RetrievalState) -> str:
        """Route to ToolNode when the last AI message requested tools."""

        messages = state.get("messages", [])
        if not messages:
            return END
        last = messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    def build_graph(self) -> Any:
        builder = StateGraph(RetrievalState)
        builder.add_node("orchestrator", self.orchestrator)
        builder.add_node("tools", ToolNode(TOOLS))
        builder.set_entry_point("orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            self.should_continue,
            {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "orchestrator")
        return builder.compile(checkpointer=self.checkpointer)

    async def run(
        self,
        session_id: str,
        user_query: str,
    ) -> dict[str, Any]:
        """Invoke one retrieval-agent turn."""

        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.graph_recursion_limit,
            "max_concurrency": settings.graph_max_concurrency,
        }
        if self.callbacks:
            config["callbacks"] = self.callbacks

        return await self.graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]},
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
        if self.callbacks:
            config["callbacks"] = self.callbacks

        return await self.graph.ainvoke(
            Command(resume=value),
            config=config,
        )

    def get_state(self, session_id: str) -> Any:
        return self.graph.get_state({"configurable": {"thread_id": session_id}})

    async def close(self) -> None:
        if self._postgres_context is not None:
            await self._postgres_context.__aexit__(None, None, None)

