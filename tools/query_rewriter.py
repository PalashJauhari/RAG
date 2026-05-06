from langchain.tools import ToolRuntime, tool

from config.settings import settings
from middleware.llm_client import get_openai_client
from output_validation.query_rewriter import QueryRewriteResult
from prompts.query_rewriter import SYSTEM_PROMPT


client = get_openai_client()


@tool
async def query_rewriter(input_query: str, runtime: ToolRuntime) -> dict:
    """Rewrite a follow-up or ambiguous user query into a standalone retrieval query."""

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
                    f"Input query to rewrite:\n{input_query}"
                ),
            },
        ],
        text_format=QueryRewriteResult,
    )
    return response.output_parsed.model_dump()
