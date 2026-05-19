"""Long-context compaction: summarize evicted turns and emit RemoveMessage ops.

Used by ``query_normalisation_node`` when enabled. Currently disabled in the shipped graph
(``graph/graph.py`` keeps full message history) but safe to turn on for very long threads.
"""

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from config.settings import settings
from middleware.llm_client import get_llm_client
from prompts.context_editing import SUMMARY_SYSTEM_PROMPT
from tool_wrappers.prompt_plain import messages_to_plain_context


def estimate_tokens(messages: list) -> int:
    """Rough token count from plain-text message context (chars // 4).

    Avoids serializing LangChain ``response_metadata`` into token estimates.
    """
    if not messages:
        return 0
    plain = messages_to_plain_context(messages)
    return len(plain) // 4


def find_safe_truncation_point(messages: list, keep: int) -> int:
    """Choose a cut index that keeps the last ``keep`` messages from a HumanMessage boundary.

    Prevents splitting assistant tool-call sequences mid-turn when evicting old history.

    Args:
        messages: Full checkpoint message list.
        keep: Target number of recent messages to retain.

    Returns:
        Index at which to slice (messages[:cut] are evicted).
    """
    candidate = max(0, len(messages) - keep)
    while candidate < len(messages):
        message = messages[candidate]
        if isinstance(message, HumanMessage):
            break
        candidate += 1
    return candidate


async def summarize_evicted(previous_summary: str, messages_to_evict: list) -> str:
    """Merge evicted messages into the rolling conversation summary via LLM.

    Args:
        previous_summary: Summary from prior truncation passes.
        messages_to_evict: LangChain messages being removed from checkpoint.

    Returns:
        Updated summary string stored in ``message_summary`` state.
    """
    llm = get_llm_client(model=settings.openai_summary_model, temperature=0)
    prompt_messages = [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)]
    if previous_summary:
        prompt_messages.append(HumanMessage(content=f"Previous summary:\n{previous_summary}"))
    prompt_messages.extend(messages_to_evict)
    prompt_messages.append(HumanMessage(content="Update the running summary."))

    response = await llm.ainvoke(prompt_messages)
    return str(response.content)


async def truncate_and_summarize(
    messages: list,
    previous_summary: str,
) -> tuple[str, list, list[RemoveMessage]]:
    """Evict old messages when over token threshold; return summary and RemoveMessage ops.

    Args:
        messages: Current thread messages.
        previous_summary: Existing ``message_summary`` state value.

    Returns:
        Tuple of (updated_summary, kept_messages, remove_ops for LangGraph merge).
    """
    token_estimate = estimate_tokens(messages)
    keep = settings.message_summary_keep_recent
    if token_estimate <= settings.message_summary_token_threshold or len(messages) <= keep:
        return previous_summary, messages, []

    cut = find_safe_truncation_point(messages, keep)
    to_evict = messages[:cut]
    updated_summary = await summarize_evicted(previous_summary, to_evict)
    remove_ops = [RemoveMessage(id=message.id) for message in to_evict]
    return updated_summary, messages[cut:], remove_ops
