from typing import Literal

from pydantic import BaseModel, Field


EvaluationStatus = Literal["sufficient", "insufficient_recall", "intent_mismatch"]


class InformationEvaluation(BaseModel):
    """Evidence judgment produced by the information evaluator node."""

    evaluation_status: EvaluationStatus = Field(
        description=(
            "sufficient when documents answer the parsed queries; insufficient_recall when "
            "documents are relevant but incomplete; intent_mismatch when documents are mostly "
            "about a different intent or entity."
        ),
        examples=["insufficient_recall"],
    )
    missing_evidence_details: list[str] = Field(
        default_factory=list,
        description=(
            "Required and non-empty only for insufficient_recall. Each item should explain what "
            "the current documents cover and what exact evidence is still missing."
        ),
        examples=[
            [
                "Retrieved passages describe general refund policy and timelines but do not name "
                "Enterprise vs Consumer tier-specific windows; need explicit per-tier deadlines.",
            ]
        ],
    )
    evaluation_explanation: str = Field(
        description="Brief observable explanation of the evaluator decision.",
        examples="The retrieved passages are relevant to refunds but do not include Consumer tier rules.",
    )
