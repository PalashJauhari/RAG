from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from langchain_core.messages import HumanMessage, SystemMessage

from middleware.llm_client import get_llm_client
from output_validation.query_expansion import QueryExpansionResult
from prompts.query_expansion import SYSTEM_PROMPT

from pydantic import BaseModel, Field


class QueryExpansionInput(BaseModel):
    input_query: str = Field(
        description="The focused semantic query that requires vocabulary bridging. "
        "Do not alter the core intent, but prepare it for expansion with domain synonyms.",
        examples=["database performance optimization techniques"],
    )


@tool("query_expansion", args_schema=QueryExpansionInput)
async def query_expansion(input_query: str, runtime: ToolRuntime) -> dict:
    """
    [ROUTING INTENT: RECALL OPTIMIZATION]
    Use WHEN: The query is highly specific or uses strict terminology, risking missing relevant documents due to vocabulary mismatch.
    Goal: Broaden the search surface by automatically generating and injecting domain-specific synonyms and parallel terms.
    """

    messages = runtime.state.get("messages", [])
    summary = runtime.state.get("message_summary", "")

    context = (
        "## Conversation Summary\n"
        f"{summary or '(none)'}\n\n"
        "## Recent Messages\n"
        f"{messages}\n\n"
        f"Input query to expand:\n{input_query}\n\n"
        "Use this summary only as conversation context. Retrieve documents before answering."
    )

    llm = get_llm_client(output_schema=QueryExpansionResult)
    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = await llm.ainvoke(prompt_messages)
    return response.model_dump()
