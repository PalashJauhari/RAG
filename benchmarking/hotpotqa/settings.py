"""HotpotQA benchmark settings (separate from app ``config.settings``).

Loads ``benchmarking/hotpotqa/.env``. Extends :class:`~config.settings.Settings` with
dataset paths and experiment output locations under ``data/processed`` and ``data/results``.
"""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from config.settings import Settings


HOTPOTQA_ROOT = Path(__file__).resolve().parent


class HotpotQASettings(Settings):
    """Benchmark configuration: dataset knobs, paths, and RAGAS model name."""

    hotpotqa_dataset_name: str = "hotpotqa/hotpot_qa"
    hotpotqa_dataset_config: str = "fullwiki"
    hotpotqa_split: str = "validation"
    hotpotqa_max_questions: int = 500
    hotpotqa_upload_batch_size: int = 64
    hotpotqa_ragas_model: str = "gpt-4o-mini"
    hotpotqa_experiment_name: str = "default_experiment"

    # --- Artifact paths (under benchmarking/hotpotqa/data/) ---
    processed_dataset_path: Path = HOTPOTQA_ROOT / "data" / "processed" / "hotpotqa_eval.json"

    @property
    def results_dir(self) -> Path:
        """Per-experiment output directory."""
        return HOTPOTQA_ROOT / "data" / "results" / self.hotpotqa_experiment_name

    @property
    def retrieval_results_path(self) -> Path:
        """Per-experiment retrieval eval JSON from ``run_retrieval_eval``."""
        return self.results_dir / "retrieval_results.json"

    @property
    def run_metadata_path(self) -> Path:
        """Run metadata JSON written alongside retrieval results."""
        return self.results_dir / "run_metadata.json"

    @property
    def ragas_results_path(self) -> Path:
        """Per-experiment RAGAS scores JSON from ``ragas_metrics``."""
        return self.results_dir / "ragas_results.json"

    @property
    def benchmark_report_path(self) -> Path:
        """Human-readable Markdown report aggregating all metrics."""
        return self.results_dir / "benchmark_report.md"

    model_config = SettingsConfigDict(
        env_file=HOTPOTQA_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
