from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from config.settings import settings
from middleware.llm_client import get_llm_client
from observability.langfuse_handler import get_observe
from prompts.context_editing import SUMMARY_SYSTEM_PROMPT
from tool_wrappers.prompt_plain import messages_to_plain_context

observe = get_observe()


def estimate_tokens(messages: list) -> int:
    """Rough token estimate from plain context (no LC metadata dumps)."""

    if not messages:
        return 0
    plain = messages_to_plain_context(messages)
    return len(plain) // 4


def find_safe_truncation_point(messages: list, keep: int) -> int:
    """Keep the recent window from a human or non-tool-calling AI boundary."""

    candidate = max(0, len(messages) - keep)
    while candidate < len(messages):
        message = messages[candidate]
        if isinstance(message, HumanMessage):
            break
        candidate += 1
    return candidate


async def summarize_evicted(previous_summary: str, messages_to_evict: list) -> str:
    """Merge old messages into the running conversation summary."""

    llm = get_llm_client(model=settings.openai_summary_model, temperature=0)
    prompt_messages = [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)]
    if previous_summary:
        prompt_messages.append(HumanMessage(content=f"Previous summary:\n{previous_summary}"))
    prompt_messages.extend(messages_to_evict)
    prompt_messages.append(HumanMessage(content="Update the running summary."))

    response = await llm.ainvoke(prompt_messages)
    return str(response.content)


@observe(name="truncate_and_summarize")
async def truncate_and_summarize(
    messages: list,
    previous_summary: str,
) -> tuple[str, list, list[RemoveMessage]]:
    """Return updated summary, kept messages, and RemoveMessage updates."""

    token_estimate = estimate_tokens(messages)
    keep = settings.message_summary_keep_recent
    if token_estimate <= settings.message_summary_token_threshold or len(messages) <= keep:
        return previous_summary, messages, []

    cut = find_safe_truncation_point(messages, keep)
    to_evict = messages[:cut]
    updated_summary = await summarize_evicted(previous_summary, to_evict)
    remove_ops = [RemoveMessage(id=message.id) for message in to_evict]
    return updated_summary, messages[cut:], remove_ops
