"""Shared literals and tier ordering for per-request retrieval strategy."""

from __future__ import annotations

from typing import Literal

RetrievalStrategy = Literal[
    "fast_retrieval",
    "fast_bm25_retrieval",
    "keyword",
    "fast_bm25_late_interaction_retrieval",
]

# Canonical low → high order for strategy_upgrade enforcement (see README).
RETRIEVAL_STRATEGY_ORDER: tuple[RetrievalStrategy, ...] = (
    "fast_retrieval",
    "keyword",
    "fast_bm25_retrieval",
    "fast_bm25_late_interaction_retrieval",
)

_STRATEGY_RANK: dict[str, int] = {
    name: index for index, name in enumerate(RETRIEVAL_STRATEGY_ORDER)
}


def retrieval_strategy_rank(strategy: str) -> int:
    """Return tier rank; unknown or empty strings rank below the lowest defined tier."""

    key = (strategy or "").strip()
    if not key:
        return -1
    return _STRATEGY_RANK.get(key, -1)


def is_strictly_heavier(current: str, proposed: str) -> bool:
    """True when ``proposed`` is a defined tier strictly above ``current``."""

    return retrieval_strategy_rank(proposed) > retrieval_strategy_rank(current)


def minimum_heavier_tier(current: str) -> RetrievalStrategy | None:
    """Smallest tier strictly above ``current``; ``None`` if already at max."""

    current_rank = retrieval_strategy_rank(current)
    for tier in RETRIEVAL_STRATEGY_ORDER:
        if _STRATEGY_RANK[tier] > current_rank:
            return tier
    return None
