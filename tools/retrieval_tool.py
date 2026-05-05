from langchain.tools import tool

from config.settings import settings
from retriever.retriever import Retriever


retriever = Retriever(settings)


@tool
async def retrieval_tool(queries: list[str], top_k: int | None = None) -> dict:
    """Retrieve top documents for one or more prepared search queries."""

    docs = await retriever.retrieve(queries, top_k=top_k)
    return {"documents": docs}

