"""Download HotpotQA distractor validation and write ``data/processed/hotpotqa_eval.json``."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Any

from datasets import load_dataset

from benchmarking.hotpotqa.benchmark_config import load_benchmark_env

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
SPLIT = "validation"
SUBSAMPLE_SEED = 42


def build_records(dataset) -> list[dict[str, Any]]:
    """Convert HF rows into benchmark JSON records."""

    records: list[dict[str, Any]] = []
    for row in dataset:
        support = row["supporting_facts"]
        support_by_title: dict[str, set[int]] = {}
        for title, sent_id in zip(support["title"], support["sent_id"]):
            support_by_title.setdefault(title, set()).add(sent_id)

        contexts = []
        for index, (title, sentences) in enumerate(
            zip(row["context"]["title"], row["context"]["sentences"])
        ):
            supporting_sentence_ids = sorted(support_by_title.get(title, set()))
            text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
            contexts.append(
                {
                    "context_id": f"{row['id']}:{index}",
                    "title": title,
                    "sentences": sentences,
                    "text": text,
                    "is_supporting": bool(supporting_sentence_ids),
                    "supporting_sentence_ids": supporting_sentence_ids,
                }
            )

        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "supporting_facts": [
                    [title, sent_id]
                    for title, sent_id in zip(support["title"], support["sent_id"])
                ],
                "contexts": contexts,
            }
        )
    return records


def subsample_dataset(dataset, max_questions: int):
    """Stratified subsample by (type, level) with fixed seed."""

    if max_questions <= 0:
        return dataset

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        groups[(row["type"], row["level"])].append(index)

    keys = sorted(groups.keys())
    random.seed(SUBSAMPLE_SEED)
    for key in keys:
        random.shuffle(groups[key])

    per_group = max_questions // len(keys)
    remainder = max_questions % len(keys)
    selected: list[int] = []
    for index, key in enumerate(keys):
        take = per_group + (1 if index < remainder else 0)
        selected.extend(groups[key][:take])
    return dataset.select(sorted(selected))


def main() -> None:
    """Download distractor validation, subsample, and write processed JSON."""

    parser = argparse.ArgumentParser(description="Download and process HotpotQA distractor eval set")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap questions (default: HOTPOTQA_MAX_QUESTIONS from hotpotqa/.env)",
    )
    args = parser.parse_args()

    config = load_benchmark_env()
    max_questions = args.max_questions if args.max_questions is not None else config.hotpotqa_max_questions

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
    if max_questions > 0:
        dataset = subsample_dataset(dataset, max_questions)

    records = build_records(dataset)
    config.processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    config.processed_dataset_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {config.processed_dataset_path}")


if __name__ == "__main__":
    main()
