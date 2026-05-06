from pydantic import BaseModel, Field


class AskUserInput(BaseModel):
    question: str = Field(description="Clarifying question to present to the user.")

