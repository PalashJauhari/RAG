from langchain.tools import ToolRuntime, tool

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from middleware.llm_client import get_llm_client
from output_validation.query_splitter import QuerySplitResult
from prompts.query_splitter import SYSTEM_PROMPT

@tool
async def query_splitter(input_query: str, runtime: ToolRuntime) -> dict:
    """Split a broad or multi-part query into focused retrieval queries."""

    messages = runtime.state.get("messages", [])
    summary = runtime.state.get("message_summary", "")

    llm = get_llm_client(output_schema=QuerySplitResult)
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt_messages.append(HumanMessage(content=f"Conversation Summary:\n{summary}"))
    prompt_messages.extend(messages)
    prompt_messages.append(HumanMessage(content=f"Input query to split:\n{input_query}"))

    response = await llm.ainvoke(prompt_messages)
    return response.model_dump()
