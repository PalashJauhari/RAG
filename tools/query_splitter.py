from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from clients.llm_client import get_openai_client
from config.settings import settings
from prompts.query_splitter import SYSTEM_PROMPT


class QuerySplitResult(BaseModel):
    queries: list[str] = Field(description="Focused retrieval queries.")


client = get_openai_client()


@tool
async def query_splitter(input_query: str, runtime: ToolRuntime) -> dict:
    """Split a broad or multi-part query into focused retrieval queries."""

    messages = runtime.state.get("messages", [])
    history = "\n".join(
        f"{getattr(message, 'type', 'message')}: {getattr(message, 'content', message)}"
        for message in messages[-10:]
    )
    response = await client.responses.parse(
        model=settings.openai_llm_model,
        temperature=settings.openai_temperature,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history}\n\n"
                    f"Input query to split:\n{input_query}"
                ),
            },
        ],
        text_format=QuerySplitResult,
    )
    return response.output_parsed.model_dump()
