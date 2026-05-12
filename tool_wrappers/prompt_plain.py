"""Plain-text conversation lines for prompts (no response_metadata / usage_metadata via repr)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _content_plain(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(json.dumps(block, default=str))
        return "\n".join(parts).strip()
    return str(content).strip()


def messages_to_plain_context(messages: list[Any]) -> str:
    """Format messages without dumping LangChain metadata (tokens, models, lc_run ids)."""

    chunks: list[str] = []
    for m in messages:
        if not isinstance(m, BaseMessage):
            chunks.append(str(m))
            continue
        text = _content_plain(getattr(m, "content", None))
        if isinstance(m, HumanMessage):
            chunks.append(f"Human:\n{text}".strip())
        elif isinstance(m, AIMessage):
            lines: list[str] = []
            if text:
                lines.append(f"Assistant:\n{text}".strip())
            tcs = list(getattr(m, "tool_calls", None) or [])
            if tcs:
                lines.append("Assistant tool_calls:\n" + json.dumps(tcs, ensure_ascii=False, default=str))
            chunks.append("\n".join(lines) if lines else "Assistant:")
        elif isinstance(m, ToolMessage):
            nm = getattr(m, "name", "") or "tool"
            chunks.append(f"Tool ({nm}):\n{text}".strip())
        else:
            chunks.append(f"{type(m).__name__}:\n{text}".strip())
    return "\n\n".join(chunks) if chunks else "(no messages)"
