# HotpotQA Benchmark

Local retrieval benchmark for HotpotQA fullwiki validation.

This benchmark tests the `Retriever` class directly. It does not use the LangGraph agent.

## Dataset

Source:

```text
hotpotqa/hotpot_qa
config: fullwiki
split: validation
```

Default size:

```env
HOTPOTQA_MAX_QUESTIONS=3500
```

We use validation/dev data because it includes `reference_answer`, `supporting_facts`, and paragraph contexts. The official test split is not used for local scoring because it does not include answers.

## Env

Benchmark config is separate from the app config:

```text
benchmarking/hotpotqa/.env
benchmarking/hotpotqa/.env.example
```

The Qdrant vector flags and vector names work the same way as the main app:

```env
USE_BM25=true
USE_LATE_INTERACTION=true
USE_MMR=true

QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_BM25_VECTOR_NAME=bm25
QDRANT_COLBERT_VECTOR_NAME=colbert
```

## Flow

Prepare normalized evaluation JSON:

```bash
python -m benchmarking.hotpotqa.dataset.prepare_eval_data
```

Output:

```text
benchmarking/hotpotqa/data/processed/hotpotqa_eval.json
```

Upload paragraph contexts to Qdrant:

```bash
python -m benchmarking.hotpotqa.qdrant_upload.upload
```

This creates the collection if it does not exist. It does not delete or recreate an existing collection.

Run retrieval evaluation:

```bash
python -m benchmarking.hotpotqa.evaluation.run_retrieval_eval
```

Output:

```text
benchmarking/hotpotqa/data/results/retrieval_results.json
```

Run RAGAS metrics:

```bash
python -m benchmarking.hotpotqa.metrics.ragas_metrics
```

Output:

```text
benchmarking/hotpotqa/data/results/ragas_results.json
```

## Result Fields

Prepared dataset records contain:

```json
{
  "id": "...",
  "question": "...",
  "reference_answer": "...",
  "supporting_facts": [["Title", 0]],
  "contexts": [
    {
      "context_id": "...",
      "title": "...",
      "text": "...",
      "is_supporting": true,
      "supporting_sentence_ids": [0]
    }
  ]
}
```

Retrieval results contain:

```json
{
  "id": "...",
  "question": "...",
  "reference_answer": "...",
  "reference_contexts": ["..."],
  "reference_context_ids": ["..."],
  "retrieved_contexts": ["..."],
  "retrieved_context_ids": ["..."],
  "retrieved_docs": []
}
```

## Metrics

RAGAS metrics:

- context precision
- context recall
- optional faithfulness

Faithfulness needs a response. If `HOTPOTQA_INCLUDE_FAITHFULNESS=true`, the script uses `generated_answer` when present in retrieval results, otherwise it uses `reference_answer` and records that source.

