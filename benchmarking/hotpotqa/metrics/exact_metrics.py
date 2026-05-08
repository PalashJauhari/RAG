import asyncio
import json
from statistics import mean

from benchmarking.hotpotqa.settings import HotpotQASettings


async def main() -> None:
    settings = HotpotQASettings()
    rows = json.loads(settings.retrieval_results_path.read_text(encoding="utf-8"))

    scored_rows = []
    for index, row in enumerate(rows, start=1):
        reference_ids = set(row["reference_context_ids"])
        retrieved_ids = row["retrieved_context_ids"]  # Ordered by rank from retriever
        
        # Calculate exact ID matches
        hits = set(reference_ids) & set(retrieved_ids)
        
        recall = len(hits) / len(reference_ids) if reference_ids else 0.0
        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
        
        # Hit Rate: 1 if at least one relevant document is retrieved, 0 otherwise
        hit_rate = 1.0 if hits else 0.0
        
        # MRR (Mean Reciprocal Rank): 1/rank of the *first* relevant document found
        mrr = 0.0
        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in reference_ids:
                mrr = 1.0 / rank
                break
                
        scored = {
            "id": row["id"],
            "question": row["question"],
            "type": row.get("type", "unknown"),
            "level": row.get("level", "unknown"),
            "exact_precision": precision,
            "exact_recall": recall,
            "hit_rate": hit_rate,
            "mrr": mrr,
        }

        scored_rows.append(scored)

    if not scored_rows:
        raise ValueError("No retrieval results found to score")

    summary = {
        "total_scored": len(scored_rows),
        "mean_exact_precision": mean(row["exact_precision"] for row in scored_rows),
        "mean_exact_recall": mean(row["exact_recall"] for row in scored_rows),
        "mean_hit_rate": mean(row["hit_rate"] for row in scored_rows),
        "mean_mrr": mean(row["mrr"] for row in scored_rows),
    }

    output = {"summary": summary, "rows": scored_rows}
    
    # Save to a new JSON file
    exact_results_path = settings.ragas_results_path.parent / "exact_metrics.json"
    exact_results_path.parent.mkdir(parents=True, exist_ok=True)
    exact_results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    print("=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote exact ID-based results to {exact_results_path}")

    # Generate Excel summary
    import pandas as pd

    df = pd.DataFrame(scored_rows)
    grouped = df.groupby(["type", "level"]).agg({
        "id": "count",
        "exact_precision": "mean",
        "exact_recall": "mean",
        "hit_rate": "mean",
        "mrr": "mean",
    }).rename(columns={"id": "count"}).reset_index()

    excel_path = settings.ragas_results_path.parent / "exact_type_level_metrics.xlsx"
    grouped.to_excel(excel_path, index=False)
    print(f"Wrote type x level Excel summary to {excel_path}")


if __name__ == "__main__":
    asyncio.run(main())
