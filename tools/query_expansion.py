from langchain.tools import ToolRuntime, tool

from config.settings import settings
from middleware.llm_client import get_openai_client
from output_validation.query_expansion import QueryExpansionResult
from prompts.query_expansion import SYSTEM_PROMPT


client = get_openai_client()


@tool
async def query_expansion(input_query: str, runtime: ToolRuntime) -> dict:
    """Expand a query with retrieval-friendly context while preserving intent."""

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
                    f"Input query to expand:\n{input_query}"
                ),
            },
        ],
        text_format=QueryExpansionResult,
    )
    return response.output_parsed.model_dump()
