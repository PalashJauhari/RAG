"""Structured output for ``strategy_upgrade_node`` (prompt: ``prompts/strategy_upgrade.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from output_validation.retrieval_strategy import (
    RetrievalStrategy,
    is_strictly_heavier,
    minimum_heavier_tier,
)


class StrategyUpgradeResult(BaseModel):
    """Whether to bump retrieval tier before the next retrieval pass."""

    apply_strategy_upgrade: bool = Field(
        description=(
            "True when gap-fill queries are sound but the current retrieval tier is too weak."
        ),
    )
    next_retrieval_strategy: RetrievalStrategy | None = Field(
        default=None,
        description="Required when apply_strategy_upgrade is true: strictly heavier tier.",
    )
    upgrade_explanation: str = Field(
        description="Brief explanation of the tier decision.",
    )

    @model_validator(mode="after")
    def validate_upgrade_fields(self) -> "StrategyUpgradeResult":
        if self.apply_strategy_upgrade and self.next_retrieval_strategy is None:
            raise ValueError(
                "next_retrieval_strategy is required when apply_strategy_upgrade is true"
            )
        if not self.apply_strategy_upgrade and self.next_retrieval_strategy is not None:
            raise ValueError(
                "next_retrieval_strategy must be omitted unless apply_strategy_upgrade is true"
            )
        return self


def resolve_strategy_upgrade(
    current_tier: str,
    result: StrategyUpgradeResult,
) -> StrategyUpgradeResult:
    """Clamp invalid tier jumps; clear apply flag when already at max tier."""

    if not result.apply_strategy_upgrade or result.next_retrieval_strategy is None:
        return result

    proposed = result.next_retrieval_strategy
    current = (current_tier or "").strip()
    if is_strictly_heavier(current, proposed):
        return result

    heavier = minimum_heavier_tier(current)
    if heavier is None:
        note = (
            "[validation] retrieval tier already at maximum "
            "(fast_bm25_late_interaction_retrieval); cannot upgrade further."
        )
        explanation = result.upgrade_explanation.strip()
        if note not in explanation:
            explanation = f"{explanation} {note}".strip() if explanation else note
        return result.model_copy(
            update={"apply_strategy_upgrade": False, "upgrade_explanation": explanation}
        )

    note = f"[validation] clamped upgrade tier from {proposed!r} to {heavier!r}."
    explanation = result.upgrade_explanation.strip()
    if note not in explanation:
        explanation = f"{explanation} {note}".strip() if explanation else note
    return result.model_copy(
        update={
            "next_retrieval_strategy": heavier,
            "upgrade_explanation": explanation,
        }
    )
