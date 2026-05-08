from langchain.tools import ToolRuntime, tool

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from middleware.llm_client import get_llm_client
from output_validation.query_expansion import QueryExpansionResult
from prompts.query_expansion import SYSTEM_PROMPT




from pydantic import BaseModel, Field

class QueryExpansionInput(BaseModel):
    input_query: str = Field(
        description="The focused semantic query that requires vocabulary bridging. "
                    "Do not alter the core intent, but prepare it for expansion with domain synonyms.",
        examples=["database performance optimization techniques"]
    )

@tool(args_schema=QueryExpansionInput, name="query_expansion")
async def query_expansion(input_query: str, runtime: ToolRuntime) -> dict:
    """
    [ROUTING INTENT: RECALL OPTIMIZATION]
    Use WHEN: The query is highly specific or uses strict terminology, risking missing relevant documents due to vocabulary mismatch.
    Goal: Broaden the search surface by automatically generating and injecting domain-specific synonyms and parallel terms.
    """

    messages = runtime.state.get("messages", [])
    summary = runtime.state.get("message_summary", "")

    llm = get_llm_client(output_schema=QueryExpansionResult)
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt_messages.append(HumanMessage(content=f"Conversation Summary:\n{summary}"))
    prompt_messages.extend(messages)
    prompt_messages.append(HumanMessage(content=f"Input query to expand:\n{input_query}"))

    response = await llm.ainvoke(prompt_messages)
    return response.model_dump()
