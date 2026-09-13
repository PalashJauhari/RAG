"""HotpotQA CLI PQ parsing, collection names, and recall-only reports."""

from __future__ import annotations

import json

import pytest

from benchmarking.hotpotqa.benchmark_config import (
    collection_name_for_pq,
    load_benchmark_config,
    parse_pq_list,
)
from benchmarking.hotpotqa.run_evaluation import write_benchmark_report


def test_parse_pq_list_default_and_dedupe() -> None:
    assert parse_pq_list(None) == ["none"]
    assert parse_pq_list([]) == ["none"]
    assert parse_pq_list(["none", "pq8", "pq16", "pq32"]) == ["none", "pq8", "pq16", "pq32"]
    assert parse_pq_list(["pq8,pq16", "none"]) == ["pq8", "pq16", "none"]
    assert parse_pq_list(["PQ16", "pq16"]) == ["pq16"]
    with pytest.raises(ValueError):
        parse_pq_list(["pq64"])


def test_collection_name_for_pq() -> None:
    assert collection_name_for_pq("hotpotqa_eval", "none") == "hotpotqa_eval_none"
    assert collection_name_for_pq("hotpotqa_eval", "pq8") == "hotpotqa_eval_pq8"
    with pytest.raises(ValueError):
        collection_name_for_pq("", "none")


def test_results_dir_is_experiment_then_pq() -> None:
    config = load_benchmark_config(
        "exp1",
        dense_pq="pq8",
        collection_base="hotpotqa_eval",
        max_questions=10,
        ragas_model="gpt-4o-mini",
    )
    assert config.qdrant_collection_name == "hotpotqa_eval_pq8"
    assert config.results_dir.name == "pq8"
    assert config.results_dir.parent.name == "exp1"
    assert config.benchmark_report_path.name == "benchmark_report.md"


def test_write_benchmark_report_recall_and_latency_only(tmp_path) -> None:
    ragas_path = tmp_path / "ragas_results.json"
    report_path = tmp_path / "benchmark_report.md"
    ragas_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_scored": 2,
                    "skipped_empty_contexts": 0,
                    "mean_context_recall": 0.75,
                },
                "rows": [
                    {"type": "bridge", "level": "hard", "context_recall": 1.0},
                    {"type": "comparison", "level": "hard", "context_recall": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )

    class _Cfg:
        hotpotqa_experiment_name = "exp_md"
        dense_pq = "none"
        qdrant_collection_name = "hotpot_base_none"
        ragas_results_path = ragas_path
        benchmark_report_path = report_path
        results_path = tmp_path / "results.json"
        run_metadata_path = tmp_path / "run_metadata.json"

    metadata = {
        "retrieval_strategy": "fast_bm25_retrieval",
        "question_count": 2,
        "retrieval_top_k": 5,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "latency": {
            "mean_ms": 10.0,
            "p50_ms": 9.0,
            "p95_ms": 12.0,
            "min_ms": 8.0,
            "max_ms": 12.0,
        },
    }
    write_benchmark_report(_Cfg(), "retrieval", metadata)
    text = report_path.read_text(encoding="utf-8")
    assert "Context recall" in text
    assert "0.7500" in text
    assert "Retriever latency" in text
    assert "Context precision" not in text
    assert "Faithfulness" not in text
    assert "Answer correctness" not in text
    assert "hotpot_base_none" in text
