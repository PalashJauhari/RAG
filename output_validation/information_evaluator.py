from typing import Literal

from pydantic import BaseModel, Field, model_validator

from output_validation.retrieval_strategy import RetrievalStrategy


EvaluationStatus = Literal[
    "sufficient",
    "insufficient_recall",
    "intent_mismatch",
    "strategy_upgrade",
]


class InformationEvaluation(BaseModel):
    """Evidence judgment produced by the information evaluator node."""

    evaluation_status: EvaluationStatus = Field(
        description=(
            "sufficient when documents answer the question; insufficient_recall when "
            "documents are relevant but incomplete; intent_mismatch when documents are mostly "
            "about a different intent or entity; strategy_upgrade when queries are sound but "
            "the current retrieval tier is too weak—request next_retrieval_strategy and reuse "
            "active_retrieval_queries without gap-fill or intent rewriting."
        ),
        examples=["insufficient_recall"],
    )
    next_retrieval_strategy: RetrievalStrategy | None = Field(
        default=None,
        description=(
            "Required when evaluation_status is strategy_upgrade: the heavier retrieval tier "
            "to apply on the next retrieval pass. Omit when status is not strategy_upgrade."
        ),
        examples=["fast_bm25_retrieval"],
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
        examples=[
            "The retrieved passages are relevant to refunds but do not include Consumer tier rules.",
        ],
    )

    @model_validator(mode="after")
    def strategy_upgrade_requires_next(self) -> "InformationEvaluation":
        if self.evaluation_status == "strategy_upgrade" and self.next_retrieval_strategy is None:
            raise ValueError("next_retrieval_strategy is required when evaluation_status is strategy_upgrade")
        if self.evaluation_status != "strategy_upgrade" and self.next_retrieval_strategy is not None:
            raise ValueError("next_retrieval_strategy must be omitted unless evaluation_status is strategy_upgrade")
        return self
