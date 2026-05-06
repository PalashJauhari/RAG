from pydantic import BaseModel, Field


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(description="The standalone retrieval-ready query.")

