"""HotpotQA benchmark paths and three local env variables only."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

HOTPOTQA_ROOT = Path(__file__).resolve().parent


class BenchmarkConfig(BaseSettings):
    """Loads ``benchmarking/hotpotqa/.env`` (max questions, experiment name, RAGAS model)."""

    hotpotqa_max_questions: int = Field(default=200, alias="HOTPOTQA_MAX_QUESTIONS")
    hotpotqa_experiment_name: str = Field(
        default="default_experiment",
        alias="HOTPOTQA_EXPERIMENT_NAME",
    )
    hotpotqa_ragas_model: str = Field(default="gpt-4o-mini", alias="HOTPOTQA_RAGAS_MODEL")

    model_config = SettingsConfigDict(
        env_file=HOTPOTQA_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def processed_dataset_path(self) -> Path:
        return HOTPOTQA_ROOT / "data" / "processed" / "hotpotqa_eval.json"

    @property
    def results_dir(self) -> Path:
        return HOTPOTQA_ROOT / "data" / "results" / self.hotpotqa_experiment_name

    @property
    def results_path(self) -> Path:
        return self.results_dir / "results.json"

    @property
    def run_metadata_path(self) -> Path:
        return self.results_dir / "run_metadata.json"

    @property
    def ragas_results_path(self) -> Path:
        return self.results_dir / "ragas_results.json"

    @property
    def benchmark_report_path(self) -> Path:
        return self.results_dir / "benchmark_report.md"


def load_benchmark_config(experiment_name: str | None = None) -> BenchmarkConfig:
    """Load config; optional experiment name overrides ``HOTPOTQA_EXPERIMENT_NAME``."""

    if experiment_name:
        return BenchmarkConfig(hotpotqa_experiment_name=experiment_name)
    return BenchmarkConfig()
