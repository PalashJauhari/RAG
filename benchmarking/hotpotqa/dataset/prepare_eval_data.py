"""Prepare normalized HotpotQA validation JSON for retrieval benchmarking.

Downloads ``hotpotqa/hotpot_qa`` (fullwiki validation), optionally subsamples with
deterministic stratified sampling (seed 42) by ``(type, level)``, and writes
``data/processed/hotpotqa_eval.json``.

Run: ``python -m benchmarking.hotpotqa.dataset.prepare_eval_data``
"""

import json
import random
from collections import defaultdict

from datasets import load_dataset

from benchmarking.hotpotqa.settings import HotpotQASettings


def main() -> None:
    """Load HF dataset, build context records with supporting-fact flags, write JSON."""
    settings = HotpotQASettings()
    dataset = load_dataset(
        settings.hotpotqa_dataset_name,
        settings.hotpotqa_dataset_config,
        split=settings.hotpotqa_split,
    )
    # --- Stratified subsample (equal type/level representation, seed 42) ---
    if settings.hotpotqa_max_questions > 0:

        groups = defaultdict(list)
        for i, row in enumerate(dataset):
            groups[(row["type"], row["level"])].append(i)

        keys = sorted(groups.keys())
        random.seed(42)
        for key in keys:
            random.shuffle(groups[key])

        target_per_group = settings.hotpotqa_max_questions // len(keys)

        selected_indices = []
        for key in keys:
            selected_indices.extend(groups[key][:target_per_group])

        dataset = dataset.select(sorted(selected_indices))

    records = []
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
                    [title, sent_id] for title, sent_id in zip(support["title"], support["sent_id"])
                ],
                "contexts": contexts,
            }
        )

    settings.processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    settings.processed_dataset_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {settings.processed_dataset_path}")


if __name__ == "__main__":
    main()

