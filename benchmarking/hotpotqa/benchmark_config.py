"""HotpotQA benchmark paths and CLI-only run config (no hotpotqa/.env)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_pq import normalize_dense_pq

HOTPOTQA_ROOT = Path(__file__).resolve().parent
PROCESSED_DATASET_PATH = HOTPOTQA_ROOT / "data" / "processed" / "hotpotqa_eval.json"

DEFAULT_MAX_QUESTIONS = 200
DEFAULT_RAGAS_MODEL = "gpt-4o-mini"


def parse_pq_list(values: list[str] | None) -> list[str]:
    """Parse CLI ``--pq`` tokens (space or comma separated). Default ``[none]``."""

    if not values:
        return ["none"]
    ordered: list[str] = []
    for raw in values:
        for part in str(raw).replace(",", " ").split():
            key = normalize_dense_pq(part)
            if key not in ordered:
                ordered.append(key)
    return ordered or ["none"]


def collection_name_for_pq(collection_base: str, dense_pq: str) -> str:
    """``{base}_{none|pq8|pq16|pq32}``."""

    base = (collection_base or "").strip()
    if not base:
        raise ValueError("--collection-base is required and must be non-empty")
    if "/" in base or "\\" in base:
        raise ValueError("--collection-base must be a single name (no slashes)")
    return f"{base}_{normalize_dense_pq(dense_pq)}"


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
    """One sequential PQ eval run: CLI experiment + pq subfolder."""

    hotpotqa_experiment_name: str
    dense_pq: str
    collection_base: str
    hotpotqa_max_questions: int = 0
    hotpotqa_ragas_model: str = DEFAULT_RAGAS_MODEL

    def __post_init__(self) -> None:
        self.hotpotqa_experiment_name = validate_experiment_name(self.hotpotqa_experiment_name)
        self.dense_pq = normalize_dense_pq(self.dense_pq)
        self.collection_base = (self.collection_base or "").strip()
        if not self.collection_base:
            raise ValueError("--collection-base is required")

    @property
    def processed_dataset_path(self) -> Path:
        return PROCESSED_DATASET_PATH

    @property
    def qdrant_collection_name(self) -> str:
        return collection_name_for_pq(self.collection_base, self.dense_pq)

    @property
    def results_dir(self) -> Path:
        return HOTPOTQA_ROOT / "data" / "results" / self.hotpotqa_experiment_name / self.dense_pq

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


def load_benchmark_config(
    experiment_name: str,
    *,
    dense_pq: str,
    collection_base: str,
    max_questions: int = 0,
    ragas_model: str = DEFAULT_RAGAS_MODEL,
) -> BenchmarkRunConfig:
    """Bind CLI flags for one PQ eval folder."""

    return BenchmarkRunConfig(
        hotpotqa_experiment_name=experiment_name,
        dense_pq=dense_pq,
        collection_base=collection_base,
        hotpotqa_max_questions=max_questions,
        hotpotqa_ragas_model=ragas_model or DEFAULT_RAGAS_MODEL,
    )
