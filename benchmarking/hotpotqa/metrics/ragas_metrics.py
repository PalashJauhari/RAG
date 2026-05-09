import asyncio
import json
from statistics import mean

from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall

from benchmarking.hotpotqa.settings import HotpotQASettings
from middleware.llm_client import get_embeddings_client


async def main() -> None:
    settings = HotpotQASettings()
    rows = json.loads(settings.retrieval_results_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_max_questions > 0:
        rows = rows[: settings.hotpotqa_max_questions]

    llm = llm_factory(
        settings.hotpotqa_ragas_model,
        client=get_embeddings_client(settings.openai_api_key),
    )
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    scored_rows = []
    for index, row in enumerate(rows, start=1):
        retrieved_contexts = row["retrieved_contexts"]
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

        scored = {
            "id": row["id"],
            "question": row["question"],
            "type": row.get("type", "unknown"),
            "level": row.get("level", "unknown"),
            "context_precision": float(precision.value),
            "context_recall": float(recall.value),
        }

        scored_rows.append(scored)
        print(f"Scored {index}/{len(rows)} questions")

    if not scored_rows:
        raise ValueError("No retrieval results found to score")

    summary = {
        "total_scored": len(scored_rows),
        "mean_context_precision": mean(row["context_precision"] for row in scored_rows),
        "mean_context_recall": mean(row["context_recall"] for row in scored_rows),
    }

    output = {"summary": summary, "rows": scored_rows}
    settings.ragas_results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ragas_results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote RAGAS results to {settings.ragas_results_path}")

    import pandas as pd
    from pathlib import Path

    df = pd.DataFrame(scored_rows)
    grouped = df.groupby(["type", "level"]).agg({
        "id": "count",
        "context_precision": "mean",
        "context_recall": "mean"
    }).rename(columns={"id": "count"}).reset_index()

    excel_path = settings.ragas_results_path.parent / "type_level_metrics.xlsx"
    grouped.to_excel(excel_path, index=False)
    print(f"Wrote type x level Excel summary to {excel_path}")


if __name__ == "__main__":
    asyncio.run(main())
