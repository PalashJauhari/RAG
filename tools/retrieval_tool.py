from __future__ import annotations

from langchain_core.tools import tool

from config.settings import settings
from retriever.retriever import Retriever
from tool_wrappers.hotpotqa_wrapper import compact_documents_for_llm


retriever = Retriever(settings)


from pydantic import BaseModel, Field

class RetrievalToolInput(BaseModel):
    queries: list[str] = Field(
        description="An array of clean, optimized semantic search queries ready for vector search execution.",
        examples=[["machine learning model deployment latency", "strategies to reduce LLM inference time"]]
    )
    top_k: int | None = Field(
        default=None, 
        description="Maximum documents to retrieve per query. Null uses system default.",
        examples=[5, 10]
    )

@tool("retrieval_tool", args_schema=RetrievalToolInput)
async def retrieval_tool(queries: list[str], top_k: int | None = None) -> dict:
    """
    [ROUTING INTENT: EXECUTE SEARCH]
    Use WHEN: You possess one or more clean, optimized, or expanded queries ready for semantic search.
    Always execute this tool AFTER query preparation to gather actual knowledge base context.
    """

    docs = await retriever.retrieve(queries, top_k=top_k)
    return {"documents": compact_documents_for_llm(docs)}

