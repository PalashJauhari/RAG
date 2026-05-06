from pydantic import BaseModel, Field


class QuerySplitResult(BaseModel):
    queries: list[str] = Field(description="Focused retrieval queries.")

