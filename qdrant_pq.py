"""Dense product-quantization helpers shared by ingestion and retrieval.

Qdrant PQ compression: pq8 → x8, pq16 → x16, pq32 → x32.
Query-time oversampling and full-precision rescore are Qdrant search params, not a
second limit we multiply ourselves.
"""

from __future__ import annotations

from typing import Any

DENSE_PQ_VALUES = frozenset({"none", "pq8", "pq16", "pq32"})
PQ_OVERSAMPLING = 3.0


def normalize_dense_pq(value: str | None) -> str:
    """Return ``none``, ``pq8``, ``pq16``, or ``pq32``."""

    key = str(value or "none").strip().lower()
    if key in {"", "none", "false", "0", "off"}:
        return "none"
    if key not in DENSE_PQ_VALUES:
        allowed = ", ".join(sorted(DENSE_PQ_VALUES))
        raise ValueError(f"DENSE_PQ must be one of: {allowed}")
    return key


def product_quantization_config(dense_pq: str) -> Any | None:
    """Qdrant ``ProductQuantization`` for collection create, or None when PQ is off."""

    from qdrant_client import models

    key = normalize_dense_pq(dense_pq)
    if key == "none":
        return None
    compression = {
        "pq8": models.CompressionRatio.X8,
        "pq16": models.CompressionRatio.X16,
        "pq32": models.CompressionRatio.X32,
    }[key]
    config_kwargs: dict[str, Any] = {"compression": compression}
    memory = getattr(models, "Memory", None)
    if memory is not None and hasattr(memory, "PINNED"):
        config_kwargs["memory"] = memory.PINNED
    return models.ProductQuantization(
        product=models.ProductQuantizationConfig(**config_kwargs)
    )


def dense_quantization_search_params(dense_pq: str) -> Any | None:
    """Search params: quantized HNSW, oversample 3×, rescore with original vectors."""

    from qdrant_client import models

    if normalize_dense_pq(dense_pq) == "none":
        return None
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=PQ_OVERSAMPLING,
        )
    )
