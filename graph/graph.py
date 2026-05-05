from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from clients.llm_client import get_chat_llm
from prompts.orchestrator import SYSTEM_PROMPT
from tools.ask_user import ask_user
from tools.query_expansion import query_expansion
from tools.query_rewriter import query_rewriter
from tools.query_splitter import query_splitter
from tools.retrieval_tool import retrieval_tool


TOOLS = [
    query_rewriter,
    query_expansion,
    query_splitter,
    retrieval_tool,
    ask_user,
]

llm_with_tools = get_chat_llm(json_mode=True).bind_tools(TOOLS)


async def orchestrator(state: MessagesState) -> dict:
    """Decide which retrieval tools to call, then produce the final JSON answer."""

    response = await llm_with_tools.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", tools_condition)
builder.add_edge("tools", "orchestrator")

graph = builder.compile(checkpointer=InMemorySaver())
