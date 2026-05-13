from pydantic import BaseModel, Field


class GapFillResult(BaseModel):
    """Missing-evidence queries generated after insufficient recall."""

    missing_queries: list[str] = Field(
        description="Focused retrieval queries targeting the missing evidence.",
        examples=[["consumer refund policy cancellation window", "enterprise refund policy exceptions"]],
    )
    gap_fill_explanation: str = Field(
        description="Brief explanation of how the missing queries address the recall gap.",
        examples="Generated one query for each missing tier-specific refund detail.",
    )
