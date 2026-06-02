"""Orchestrate HotpotQA graph benchmark: graph eval → RAGAS → report.

Single entry point. Run from repo root with venv ``rag_env_1`` active::

    python -m benchmarking.hotpotqa.run_graph_benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import os

from benchmarking.hotpotqa.evaluation.run_graph_eval import run_graph_eval
from benchmarking.hotpotqa.metrics.build_graph_report import build_graph_report_markdown
from benchmarking.hotpotqa.metrics.ragas_graph_metrics import run_graph_ragas_metrics
from benchmarking.hotpotqa.settings import HotpotQASettings


async def run_graph_benchmark(*, experiment_name: str | None = None) -> None:
    """Run the full graph benchmark pipeline for one experiment folder.

    Steps:
    1. :func:`~benchmarking.hotpotqa.evaluation.run_graph_eval.run_graph_eval`
    2. :func:`~benchmarking.hotpotqa.metrics.ragas_graph_metrics.run_graph_ragas_metrics`
    3. :func:`~benchmarking.hotpotqa.metrics.build_graph_report.build_graph_report_markdown`

    Args:
        experiment_name: Optional override for ``HOTPOTQA_EXPERIMENT_NAME``.
    """
    if experiment_name:
        os.environ["HOTPOTQA_EXPERIMENT_NAME"] = experiment_name

    settings = HotpotQASettings()

    print(f"Experiment: {settings.hotpotqa_experiment_name}")

    await run_graph_eval(settings=settings)
    await run_graph_ragas_metrics(settings=settings)
    build_graph_report_markdown(settings=settings)
    print("Graph benchmark complete.")


def main() -> None:
    """CLI entry point for the HotpotQA graph benchmark."""

    parser = argparse.ArgumentParser(
        description="HotpotQA graph benchmark (RetrievalGraph, RAGAS, report)"
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override HOTPOTQA_EXPERIMENT_NAME for this run",
    )
    args = parser.parse_args()
    asyncio.run(run_graph_benchmark(experiment_name=args.experiment_name))


if __name__ == "__main__":
    main()
