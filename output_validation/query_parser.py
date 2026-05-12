from pydantic import BaseModel, Field


class QueryParserOutput(BaseModel):
    """Final retrieval queries prepared by the query parser node."""

    queries: list[str] = Field(
        description="Clean retrieval queries to send to the retriever. Must not be empty.",
        examples=[["enterprise refund policy", "consumer refund policy"]],
    )
    decomposition_applied: bool = Field(
        description="Whether query decomposition was applied.",
        examples=[True],
    )
    expansion_applied: bool = Field(
        description="Whether query expansion was applied.",
        examples=[False],
    )
