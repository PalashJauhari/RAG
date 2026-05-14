from pydantic import BaseModel, Field


class IntentCorrectionRewriteResult(BaseModel):
    """Corrected retrieval queries after the evaluator detects intent mismatch."""

    corrected_queries: list[str] = Field(
        description="Retrieval queries rewritten to better match the intended user request.",
        examples=[["enterprise refund policy", "consumer refund policy"]],
    )
    correction_explanation: str = Field(
        description="Brief explanation of what intent drift was corrected.",
        examples=[
            "Removed wording that pulled retrieval toward general pricing documents.",
        ],
    )
