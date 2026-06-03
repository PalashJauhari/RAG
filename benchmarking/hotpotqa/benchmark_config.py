"""HotpotQA benchmark paths and local env variables (not experiment name)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

HOTPOTQA_ROOT = Path(__file__).resolve().parent


class BenchmarkEnvSettings(BaseSettings):
    """Loads ``benchmarking/hotpotqa/.env`` (max questions, RAGAS model only)."""

    hotpotqa_max_questions: int = Field(default=200, alias="HOTPOTQA_MAX_QUESTIONS")
    hotpotqa_ragas_model: str = Field(default="gpt-4o-mini", alias="HOTPOTQA_RAGAS_MODEL")

    model_config = SettingsConfigDict(
        env_file=HOTPOTQA_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def processed_dataset_path(self) -> Path:
        return HOTPOTQA_ROOT / "data" / "processed" / "hotpotqa_eval.json"


def validate_experiment_name(experiment_name: str) -> str:
    """Normalize and validate a CLI ``--experiment-name`` value."""

    key = (experiment_name or "").strip()
    if not key:
        raise ValueError("--experiment-name is required and must be non-empty")
    if "/" in key or "\\" in key or key in {".", ".."}:
        raise ValueError(
            f"Invalid --experiment-name {experiment_name!r}: use a single path segment "
            "(no slashes or ..)."
        )
    return key


@dataclass
class BenchmarkRunConfig:
    """Eval run config: env settings plus required experiment name from CLI."""

    env: BenchmarkEnvSettings
    hotpotqa_experiment_name: str

    @property
    def hotpotqa_max_questions(self) -> int:
        return self.env.hotpotqa_max_questions

    @property
    def hotpotqa_ragas_model(self) -> str:
        return self.env.hotpotqa_ragas_model

    @property
    def processed_dataset_path(self) -> Path:
        return self.env.processed_dataset_path

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


def load_benchmark_env() -> BenchmarkEnvSettings:
    """Load HotpotQA env settings for download/upload (no experiment name)."""

    return BenchmarkEnvSettings()


def load_benchmark_config(experiment_name: str) -> BenchmarkRunConfig:
    """Load env settings and bind a required experiment name from ``--experiment-name``."""

    return BenchmarkRunConfig(
        env=load_benchmark_env(),
        hotpotqa_experiment_name=validate_experiment_name(experiment_name),
    )


# Back-compat alias for scripts that only need processed dataset paths.
BenchmarkConfig = BenchmarkEnvSettings
