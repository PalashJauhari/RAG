from pydantic import BaseModel, Field


class QueryExpansionResult(BaseModel):
    expanded_query: str = Field(description="The retrieval-ready expanded query.")
    added_context: str = Field(description="Short note on what context was added.")

