"""Dense product-quantization env mapping and search params."""

from __future__ import annotations

import pytest

from qdrant_pq import (
    dense_quantization_search_params,
    normalize_dense_pq,
    product_quantization_config,
)


def test_normalize_dense_pq() -> None:
    assert normalize_dense_pq("none") == "none"
    assert normalize_dense_pq("PQ16") == "pq16"
    assert normalize_dense_pq("") == "none"
    with pytest.raises(ValueError):
        normalize_dense_pq("pq64")


def test_none_skips_qdrant_configs() -> None:
    assert product_quantization_config("none") is None
    assert dense_quantization_search_params("none") is None


def test_pq16_sets_x16_and_rescore_oversample() -> None:
    from qdrant_client import models

    pq = product_quantization_config("pq16")
    assert pq is not None
    assert pq.product.compression == models.CompressionRatio.X16

    params = dense_quantization_search_params("pq16")
    assert params is not None
    assert params.quantization.ignore is False
    assert params.quantization.rescore is True
    assert params.quantization.oversampling == 3.0
