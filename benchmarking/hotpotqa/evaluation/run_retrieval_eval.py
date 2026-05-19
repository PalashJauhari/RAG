"""Run :class:`~retriever.retriever.Retriever` on HotpotQA questions (no LangGraph).

Picks default strategy from ``USE_LATE_INTERACTION`` / ``USE_BM25`` flags. Writes
``retrieval_results.json`` under ``data/results/{experiment}/``.

Run: ``python -m benchmarking.hotpotqa.evaluation.run_retrieval_eval``
"""

import asyncio
import json

from benchmarking.hotpotqa.settings import HotpotQASettings
from retriever.retriever import Retriever


async def main() -> None:
    """Retrieve once per question; record reference vs retrieved context ids and texts."""
    settings = HotpotQASettings()
    retriever = Retriever(settings)
    records = json.loads(settings.processed_dataset_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_max_questions > 0:
        records = records[: settings.hotpotqa_max_questions]

    # Strategy tier mirrors production defaults from env flags (not per-question complexity).
    if settings.use_late_interaction:
        default_strategy = "fast_bm25_late_interaction_retrieval"
    elif settings.use_bm25:
        default_strategy = "fast_bm25_retrieval"
    else:
        default_strategy = "fast_retrieval"

    results = []
    for index, record in enumerate(records, start=1):
        docs = await retriever.retrieve([record["question"]], strategy=default_strategy)
        reference_contexts = [
            context["text"] for context in record["contexts"] if context["is_supporting"]
        ]
        reference_context_ids = [
            context["context_id"] for context in record["contexts"] if context["is_supporting"]
        ]

        results.append(
            {
                "id": record["id"],
                "question": record["question"],
                "type": record["type"],
                "level": record["level"],
                "reference_answer": record["reference_answer"],
                "reference_contexts": reference_contexts,
                "reference_context_ids": reference_context_ids,
                "retrieved_contexts": [
                    doc["payload"]["text"]
                    for doc in docs
                    if doc.get("payload") and doc["payload"].get("text")
                ],
                "retrieved_context_ids": [
                    doc["payload"].get("context_id")
                    for doc in docs
                    if doc.get("payload") and doc["payload"].get("context_id")
                ],
                "retrieved_docs": docs,
            }
        )
        print(f"Evaluated {index}/{len(records)} questions")

    settings.retrieval_results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.retrieval_results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote retrieval results to {settings.retrieval_results_path}")


if __name__ == "__main__":
    asyncio.run(main())
