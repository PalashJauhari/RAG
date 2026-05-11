from langchain_core.tools import tool
from langgraph.types import interrupt

from pydantic import BaseModel, Field

class AskUserInput(BaseModel):
    question: str = Field(
        description="A direct, conversational clarifying question for the user. "
                    "Must explicitly state what critical context or undefined pronoun is preventing retrieval.",
        examples=["You mentioned 'it' - are you referring to the enterprise refund policy or the standard policy?"]
    )

@tool("ask_user", args_schema=AskUserInput)
def ask_user(question: str) -> str:
    """
    [ROUTING INTENT: AMBIGUITY RESOLUTION]
    Use WHEN: The query is dangerously ambiguous, lacks fundamental context, or relies on undefined pronouns (e.g., "What did he say about it?") that are NOT resolved in the conversation summary.
    This pauses the graph to get human input.
    """

    return interrupt({"question": question})
