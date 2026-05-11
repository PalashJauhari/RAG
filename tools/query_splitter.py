from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from langchain_core.messages import HumanMessage, SystemMessage

from middleware.llm_client import get_llm_client
from output_validation.query_splitter import QuerySplitResult
from prompts.query_splitter import SYSTEM_PROMPT

from pydantic import BaseModel, Field


class QuerySplitInput(BaseModel):
    input_query: str = Field(
        description="The complex, multi-hop, or comparative user query to be decomposed. "
        "Must contain conjunctions or distinct topics requiring separate retrieval paths.",
        examples=["What is the difference in pricing between AWS S3 and Azure Blob Storage?"],
    )


@tool("query_splitter", args_schema=QuerySplitInput)
async def query_splitter(input_query: str, runtime: ToolRuntime) -> dict:
    """
    [ROUTING INTENT: DECOMPOSITION]
    Use WHEN: The user asks a multi-hop question, compares multiple entities, or asks several independent questions in one prompt.
    Goal: Break the complex query into atomic, independent sub-queries to maximize vector search precision.
    """

    messages = runtime.state.get("messages", [])
    summary = runtime.state.get("message_summary", "")

    context = (
        "## Conversation Summary\n"
        f"{summary or '(none)'}\n\n"
        "## Recent Messages\n"
        f"{messages}\n\n"
        f"Input query to split:\n{input_query}\n\n"
        "Use this summary only as conversation context. Retrieve documents before answering."
    )

    llm = get_llm_client(output_schema=QuerySplitResult)
    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = await llm.ainvoke(prompt_messages)
    return response.model_dump()
