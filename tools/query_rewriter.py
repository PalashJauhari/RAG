from langchain.tools import ToolRuntime, tool

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from middleware.llm_client import get_llm_client
from output_validation.query_rewriter import QueryRewriteResult
from prompts.query_rewriter import SYSTEM_PROMPT




@tool
async def query_rewriter(input_query: str, runtime: ToolRuntime) -> dict:
    """Rewrite a follow-up or ambiguous user query into a standalone retrieval query."""

    messages = runtime.state.get("messages", [])
    summary = runtime.state.get("message_summary", "")

    llm = get_llm_client(output_schema=QueryRewriteResult)
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt_messages.append(HumanMessage(content=f"Conversation Summary:\n{summary}"))
    prompt_messages.extend(messages)
    prompt_messages.append(HumanMessage(content=f"Input query to rewrite:\n{input_query}"))

    response = await llm.ainvoke(prompt_messages)
    return response.model_dump()
