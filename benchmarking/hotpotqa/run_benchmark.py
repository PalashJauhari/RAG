"""Orchestrate HotpotQA retriever benchmark: retrieve → RAGAS → report.

Single entry point. Run::

    python -m benchmarking.hotpotqa.run_benchmark --strategy fast_bm25_retrieval
"""

from __future__ import annotations

import argparse
import asyncio
import os

from benchmarking.hotpotqa.evaluation.run_retrieval_eval import run_retrieval_eval
from benchmarking.hotpotqa.metrics.build_report import build_report_markdown
from benchmarking.hotpotqa.metrics.ragas_metrics import run_ragas_metrics
from benchmarking.hotpotqa.settings import HotpotQASettings
from benchmarking.hotpotqa.strategy import parse_strategy


async def run_benchmark(
    strategy: str,
    *,
    experiment_name: str | None = None,
) -> None:
    """Run the full benchmark pipeline for one retrieval strategy."""

    if experiment_name:
        os.environ["HOTPOTQA_EXPERIMENT_NAME"] = experiment_name

    settings = HotpotQASettings()
    parsed = parse_strategy(strategy)

    print(f"Experiment: {settings.hotpotqa_experiment_name}")
    print(f"Strategy: {parsed}")

    await run_retrieval_eval(parsed, settings=settings)
    await run_ragas_metrics(settings=settings)
    build_report_markdown(settings=settings)
    print("Benchmark complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HotpotQA retriever benchmark (retrieve, RAGAS, report)"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help=(
            "Retrieval strategy: fast_retrieval, keyword, fast_bm25_retrieval, "
            "fast_bm25_late_interaction_retrieval"
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override HOTPOTQA_EXPERIMENT_NAME for this run",
    )
    args = parser.parse_args()
    asyncio.run(
        run_benchmark(
            args.strategy,
            experiment_name=args.experiment_name,
        )
    )


if __name__ == "__main__":
    main()
