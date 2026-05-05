from pathlib import Path

from pydantic_settings import SettingsConfigDict

from config.settings import Settings


HOTPOTQA_ROOT = Path(__file__).resolve().parent


class HotpotQASettings(Settings):
    """Benchmark settings loaded from benchmarking/hotpotqa/.env."""

    hotpotqa_dataset_name: str = "hotpotqa/hotpot_qa"
    hotpotqa_dataset_config: str = "fullwiki"
    hotpotqa_split: str = "validation"
    hotpotqa_max_questions: int = 3500
    hotpotqa_upload_batch_size: int = 64
    hotpotqa_eval_max_questions: int = 3500
    hotpotqa_ragas_max_questions: int = 3500
    hotpotqa_include_faithfulness: bool = False
    hotpotqa_ragas_model: str = "gpt-4o-mini"

    processed_dataset_path: Path = HOTPOTQA_ROOT / "data" / "processed" / "hotpotqa_eval.json"
    retrieval_results_path: Path = HOTPOTQA_ROOT / "data" / "results" / "retrieval_results.json"
    ragas_results_path: Path = HOTPOTQA_ROOT / "data" / "results" / "ragas_results.json"

    model_config = SettingsConfigDict(
        env_file=HOTPOTQA_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
