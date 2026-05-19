from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from output_validation.retrieval_strategy import (
    RetrievalStrategy,
    is_strictly_heavier,
    minimum_heavier_tier,
)


EvaluationStatus = Literal[
    "sufficient",
    "insufficient_recall",
    "intent_mismatch",
    "strategy_upgrade",
]


def _non_empty_evidence_details(details: list[str]) -> list[str]:
    return [item.strip() for item in details if item and item.strip()]


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
    def validate_evaluation_fields(self) -> "InformationEvaluation":
        if self.evaluation_status == "strategy_upgrade" and self.next_retrieval_strategy is None:
            raise ValueError(
                "next_retrieval_strategy is required when evaluation_status is strategy_upgrade"
            )
        if self.evaluation_status != "strategy_upgrade" and self.next_retrieval_strategy is not None:
            raise ValueError(
                "next_retrieval_strategy must be omitted unless evaluation_status is strategy_upgrade"
            )

        evidence = _non_empty_evidence_details(self.missing_evidence_details)
        if self.evaluation_status == "insufficient_recall":
            if not evidence:
                raise ValueError(
                    "missing_evidence_details must contain at least one non-empty string "
                    "when evaluation_status is insufficient_recall"
                )
        elif evidence:
            raise ValueError(
                "missing_evidence_details must be empty unless evaluation_status is insufficient_recall"
            )

        return self


def resolve_strategy_upgrade(
    current_tier: str,
    evaluation: InformationEvaluation,
    *,
    max_upgrade_retries: int,
) -> tuple[InformationEvaluation, bool]:
    """Normalize strategy_upgrade tiers against ``current_tier``.

    Returns ``(evaluation, force_partial)``. When ``force_partial`` is True, the graph should set
    ``strategy_upgrade_retry_count`` to ``max_upgrade_retries`` so routing goes to ``partial_answer``
    without applying a new retrieval tier (already at max).
    """

    if evaluation.evaluation_status != "strategy_upgrade":
        return evaluation, False

    proposed = evaluation.next_retrieval_strategy
    if proposed is None:
        return evaluation, False

    current = (current_tier or "").strip()
    if is_strictly_heavier(current, proposed):
        return evaluation, False

    heavier = minimum_heavier_tier(current)
    if heavier is None:
        note = (
            "[validation] retrieval tier already at maximum "
            "(fast_bm25_late_interaction_retrieval); cannot upgrade further."
        )
        explanation = evaluation.evaluation_explanation.strip()
        if note not in explanation:
            explanation = f"{explanation} {note}".strip() if explanation else note
        return evaluation.model_copy(update={"evaluation_explanation": explanation}), True

    note = f"[validation] clamped upgrade tier from {proposed!r} to {heavier!r}."
    explanation = evaluation.evaluation_explanation.strip()
    if note not in explanation:
        explanation = f"{explanation} {note}".strip() if explanation else note
    return (
        evaluation.model_copy(
            update={
                "next_retrieval_strategy": heavier,
                "evaluation_explanation": explanation,
            }
        ),
        False,
    )
