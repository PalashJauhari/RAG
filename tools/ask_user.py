from langchain.tools import tool
from langgraph.types import interrupt

from output_validation.ask_user import AskUserInput


@tool(args_schema=AskUserInput)
def ask_user(question: str) -> str:
    """Ask one clarifying question and pause the graph until the user answers."""

    return interrupt({"question": question})
