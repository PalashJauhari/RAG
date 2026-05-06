import asyncio
import json
from statistics import mean

from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

from benchmarking.hotpotqa.settings import HotpotQASettings
from middleware.llm_client import get_openai_client


async def main() -> None:
    settings = HotpotQASettings()
    rows = json.loads(settings.retrieval_results_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_ragas_max_questions > 0:
        rows = rows[: settings.hotpotqa_ragas_max_questions]

    llm = llm_factory(
        settings.hotpotqa_ragas_model,
        client=get_openai_client(settings.openai_api_key),
    )
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)
    faithfulness = Faithfulness(llm=llm) if settings.hotpotqa_include_faithfulness else None

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
            "context_precision": float(precision.value),
            "context_recall": float(recall.value),
        }
        if faithfulness:
            response_source = (
                "generated_answer" if row.get("generated_answer") else "reference_answer"
            )
            score = await faithfulness.ascore(
                user_input=row["question"],
                response=row.get("generated_answer") or row["reference_answer"],
                retrieved_contexts=retrieved_contexts,
            )
            scored["faithfulness"] = float(score.value)
            scored["faithfulness_response_source"] = response_source

        scored_rows.append(scored)
        print(f"Scored {index}/{len(rows)} questions")

    if not scored_rows:
        raise ValueError("No retrieval results found to score")

    summary = {
        "total_scored": len(scored_rows),
        "mean_context_precision": mean(row["context_precision"] for row in scored_rows),
        "mean_context_recall": mean(row["context_recall"] for row in scored_rows),
    }
    if settings.hotpotqa_include_faithfulness:
        summary["mean_faithfulness"] = mean(row["faithfulness"] for row in scored_rows)

    output = {"summary": summary, "rows": scored_rows}
    settings.ragas_results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ragas_results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote RAGAS results to {settings.ragas_results_path}")


if __name__ == "__main__":
    asyncio.run(main())
