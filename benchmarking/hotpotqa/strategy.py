"""Parse and validate retrieval strategy literals for HotpotQA benchmarks.

CLI ``--strategy`` must be one of the same four strings used by the graph and
:class:`~retriever.retriever.Retriever` (no friendly aliases).
"""

from __future__ import annotations

from output_validation.retrieval_strategy import RETRIEVAL_STRATEGY_ORDER, RetrievalStrategy

from benchmarking.hotpotqa.settings import HotpotQASettings

ALLOWED_STRATEGIES: tuple[RetrievalStrategy, ...] = RETRIEVAL_STRATEGY_ORDER


def parse_strategy(value: str) -> RetrievalStrategy:
    """Return a validated ``RetrievalStrategy`` literal or raise ``ValueError``."""

    key = (value or "").strip()
    if key not in ALLOWED_STRATEGIES:
        allowed = ", ".join(ALLOWED_STRATEGIES)
        raise ValueError(f"Unknown retrieval strategy {value!r}. Choose one of: {allowed}")
    return key  # type: ignore[return-value]


def validate_strategy_env(strategy: RetrievalStrategy, settings: HotpotQASettings) -> None:
    """Ensure ``.env`` flags support the requested strategy before retrieval runs."""

    if strategy in {"keyword", "fast_bm25_retrieval", "fast_bm25_late_interaction_retrieval"}:
        if not settings.use_bm25:
            raise ValueError(
                f"Strategy {strategy!r} requires USE_BM25=true in benchmarking/hotpotqa/.env"
            )

    if strategy == "fast_bm25_late_interaction_retrieval":
        if not settings.use_late_interaction:
            raise ValueError(
                "Strategy fast_bm25_late_interaction_retrieval requires USE_LATE_INTERACTION=true"
            )
        if not (settings.jina_api_key or "").strip():
            raise ValueError(
                "Strategy fast_bm25_late_interaction_retrieval requires JINA_API_KEY in "
                "benchmarking/hotpotqa/.env"
            )

    if strategy == "fast_retrieval" and not (settings.openai_api_key or "").strip():
        raise ValueError("Strategy fast_retrieval requires OPENAI_API_KEY for dense embeddings")
