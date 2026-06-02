"""RAGAS scoring step for the HotpotQA graph benchmark (internal; no CLI).

Uses ``ragas.metrics.collections`` only (no custom metrics or prompts).
Reads ``graph_results.json``; writes ``graph_ragas_results.json`` and optional xlsx.
Called from :func:`~benchmarking.hotpotqa.run_graph_benchmark.run_graph_benchmark`.

Partial-answer rows are included in means when they have scorable fields (same as full answers).
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from benchmarking.hotpotqa.settings import HotpotQASettings


def mean_optional(values: list[float | None]) -> float | None:
    """Return mean of non-None values, or None when the list is empty."""

    present = [value for value in values if value is not None]
    if not present:
        return None
    return mean(present)


async def run_graph_ragas_metrics(settings: HotpotQASettings | None = None) -> dict[str, Any]:
    """LLM-judged context, faithfulness, and answer metrics per question.

    Skip rules per row:
    - Empty ``retrieved_contexts``: no context precision/recall or faithfulness.
    - Empty ``response``: no faithfulness or answer correctness.
    Partial answers are scored when inputs are present.

    Args:
        settings: Benchmark settings; defaults to :class:`HotpotQASettings`.

    Returns:
        Dict with ``summary`` and ``rows`` written to ``graph_ragas_results.json``.
    """
    settings = settings or HotpotQASettings()
    rows = json.loads(settings.graph_results_path.read_text(encoding="utf-8"))
    if settings.hotpotqa_max_questions > 0:
        rows = rows[: settings.hotpotqa_max_questions]

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.hotpotqa_ragas_model, client=client)
    embeddings = embedding_factory(
        "openai",
        model=settings.openai_embedding_model,
        client=client,
    )
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    answer_correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)

    scored_rows: list[dict[str, Any]] = []
    skipped_empty_contexts = 0
    skipped_empty_response = 0
    skipped_no_metrics = 0

    for index, row in enumerate(rows, start=1):
        question = row["question"]
        reference = row["reference_answer"]
        response = str(row.get("response") or "").strip()
        retrieved_contexts = row.get("retrieved_contexts") or []
        has_contexts = bool(retrieved_contexts)
        has_response = bool(response)

        if not has_contexts:
            skipped_empty_contexts += 1
        if not has_response:
            skipped_empty_response += 1

        context_precision_score: float | None = None
        context_recall_score: float | None = None
        faithfulness_score: float | None = None
        answer_correctness_score: float | None = None

        if has_contexts:
            precision = await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            recall = await context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            context_precision_score = float(precision.value)
            context_recall_score = float(recall.value)

        if has_contexts and has_response:
            faith = await faithfulness.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
            faithfulness_score = float(faith.value)

        if has_response:
            correct = await answer_correctness.ascore(
                user_input=question,
                response=response,
                reference=reference,
            )
            answer_correctness_score = float(correct.value)

        if (
            context_precision_score is None
            and context_recall_score is None
            and faithfulness_score is None
            and answer_correctness_score is None
        ):
            skipped_no_metrics += 1
            print(f"Skipped {index}/{len(rows)} (no scorable fields)")
            continue

        scored_rows.append(
            {
                "id": row["id"],
                "question": question,
                "type": row.get("type", "unknown"),
                "level": row.get("level", "unknown"),
                "is_partial": row.get("is_partial", False),
                "context_precision": context_precision_score,
                "context_recall": context_recall_score,
                "faithfulness": faithfulness_score,
                "answer_correctness": answer_correctness_score,
            }
        )
        print(f"Scored {index}/{len(rows)} questions")

    if not scored_rows:
        raise ValueError("No graph results found to score")

    summary = {
        "total_scored": len(scored_rows),
        "skipped_empty_contexts": skipped_empty_contexts,
        "skipped_empty_response": skipped_empty_response,
        "skipped_no_metrics": skipped_no_metrics,
        "partial_answer_count": sum(1 for row in rows if row.get("is_partial")),
        "mean_context_precision": mean_optional([r["context_precision"] for r in scored_rows]),
        "mean_context_recall": mean_optional([r["context_recall"] for r in scored_rows]),
        "mean_faithfulness": mean_optional([r["faithfulness"] for r in scored_rows]),
        "mean_answer_correctness": mean_optional(
            [r["answer_correctness"] for r in scored_rows]
        ),
    }

    output: dict[str, Any] = {"summary": summary, "rows": scored_rows}
    settings.graph_ragas_results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.graph_ragas_results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote graph RAGAS results to {settings.graph_ragas_results_path}")

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
                    "faithfulness": "mean",
                    "answer_correctness": "mean",
                }
            )
            .rename(columns={"id": "count"})
            .reset_index()
        )
        excel_path = settings.graph_ragas_results_path.parent / "graph_type_level_metrics.xlsx"
        grouped.to_excel(excel_path, index=False)
        print(f"Wrote type x level Excel summary to {excel_path}")
    except ImportError:
        print("pandas/openpyxl not available; skipped graph_type_level_metrics.xlsx")

    return output
