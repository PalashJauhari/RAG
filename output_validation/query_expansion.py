from pydantic import BaseModel, Field


class QueryExpansionResult(BaseModel):
    """Exploratory query expansion into multiple retrieval angles."""

    queries: list[str] = Field(
        description="Multiple focused retrieval queries covering the exploratory request.",
        examples=[
            [
                "enterprise refund policy overview",
                "consumer refund policy exceptions",
                "refund policy cancellation deadlines",
            ]
        ],
    )
    expansion_explanation: str = Field(
        description="Brief note describing the angles or vocabulary covered by the queries.",
        examples=["Covered policy overview, tier-specific rules, and cancellation deadlines."],
    )
