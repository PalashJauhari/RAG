"""messages_for_langfuse serialises LangChain messages for span input."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from observability.langfuse_handler import messages_for_langfuse


def test_messages_for_langfuse_role_and_content() -> None:
    rows = messages_for_langfuse(
        [HumanMessage(content="hi"), AIMessage(content='{"answer": "ok"}')]
    )
    assert rows == [
        {"role": "HumanMessage", "content": "hi"},
        {"role": "AIMessage", "content": '{"answer": "ok"}'},
    ]


def test_messages_for_langfuse_empty() -> None:
    assert messages_for_langfuse(None) == []
    assert messages_for_langfuse([]) == []
