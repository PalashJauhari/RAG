from langchain.tools import tool
from langgraph.types import interrupt


@tool
def ask_user(question: str) -> dict:
    """Ask the user for clarification and pause the graph until they answer."""

    answer = interrupt({"question": question})
    return {"answer": answer}

