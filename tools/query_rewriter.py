from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from langchain_core.messages import HumanMessage, SystemMessage

from middleware.llm_client import get_llm_client
from output_validation.query_rewriter import QueryRewriteResult
from prompts.query_rewriter import SYSTEM_PROMPT




from pydantic import BaseModel, Field

class QueryRewriteInput(BaseModel):
    input_query: str = Field(
        description="The ambiguous or context-dependent user query that requires co-reference resolution. "
                    "You must substitute all pronouns with explicit entities from the conversation summary.",
        examples=["What is the SLA for the Enterprise Tier?"]
    )

@tool("query_rewriter", args_schema=QueryRewriteInput)
async def query_rewriter(input_query: str, runtime: ToolRuntime) -> dict:
    """
    [ROUTING INTENT: PRECISION OPTIMIZATION]
    Use WHEN: The user's intent is clear from the conversation history, but their latest input is poorly phrased, conversational, or uses pronouns.
    Goal: Normalize the input into a standalone, explicit semantic search query.
    """

    messages = runtime.state.get("messages", [])
    summary = runtime.state.get("message_summary", "")

    transcript = "\n\n".join(str(m.content) for m in messages) if messages else "(no conversation messages)"

    llm = get_llm_client(output_schema=QueryRewriteResult)
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt_messages.append(HumanMessage(content=f"Conversation Summary:\n{summary}"))
    prompt_messages.append(HumanMessage(content=f"Conversation transcript:\n\n{transcript}"))
    prompt_messages.append(HumanMessage(content=f"Input query to rewrite:\n{input_query}"))

    response = await llm.ainvoke(prompt_messages)
    return response.model_dump()
