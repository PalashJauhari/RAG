"""Score retrieval results with vanilla RAGAS context precision and context recall.

Uses ``ragas.metrics.collections`` only (no custom metrics or prompts).
Reads ``retrieval_results.json``; writes ``ragas_results.json`` and optional xlsx.

Run: ``python -m benchmarking.hotpotqa.metrics.ragas_metrics``
"""

from __future__ import annotations

import asyncio
import json
from statistics import mean
from typing import Any

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall

from benchmarking.hotpotqa.settings import HotpotQASettings


async def run_ragas_metrics(settings: HotpotQASettings | None = None) -> dict[str, Any]:
    """LLM-judged context precision/recall per question; aggregate means in summary."""

    settings = settings or HotpotQASettings()
    rows = json.loads(settings.retrieval_results_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_max_questions > 0:
        rows = rows[: settings.hotpotqa_max_questions]

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.hotpotqa_ragas_model, client=client)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    scored_rows: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(rows, start=1):
        retrieved_contexts = row.get("retrieved_contexts") or []
        if not retrieved_contexts:
            skipped += 1
            print(f"Skipped {index}/{len(rows)} (empty contexts)")
            continue

        precision = await context_precision.ascore(
            user_input=row["question"],
            reference=row["reference_answer"],
            retrieved_contexts=retrieved_contexts,
        )
        recall = await context_recall.ascore(
            user_input=row["question"],
            reference=row["reference_answer"],
            retrieved_contexts=retrieved_contexts,
        )

        scored_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "type": row.get("type", "unknown"),
                "level": row.get("level", "unknown"),
                "context_precision": float(precision.value),
                "context_recall": float(recall.value),
            }
        )
        print(f"Scored {index}/{len(rows)} questions")

    if not scored_rows:
        raise ValueError("No retrieval results found to score")

    summary = {
        "total_scored": len(scored_rows),
        "skipped_empty_contexts": skipped,
        "mean_context_precision": mean(r["context_precision"] for r in scored_rows),
        "mean_context_recall": mean(r["context_recall"] for r in scored_rows),
    }

    output: dict[str, Any] = {"summary": summary, "rows": scored_rows}
    settings.ragas_results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ragas_results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote RAGAS results to {settings.ragas_results_path}")

    try:
        import pandas as pd

        df = pd.DataFrame(scored_rows)
        grouped = (
            df.groupby(["type", "level"])
            .agg(
                {
                    "id": "count",
                    "context_precision": "mean",
                    "context_recall": "mean",
                }
            )
            .rename(columns={"id": "count"})
            .reset_index()
        )
        excel_path = settings.ragas_results_path.parent / "type_level_metrics.xlsx"
        grouped.to_excel(excel_path, index=False)
        print(f"Wrote type x level Excel summary to {excel_path}")
    except ImportError:
        print("pandas/openpyxl not available; skipped type_level_metrics.xlsx")

    return output


async def main() -> None:
    await run_ragas_metrics()


if __name__ == "__main__":
    asyncio.run(main())
