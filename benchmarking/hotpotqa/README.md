# HotpotQA benchmark (distractor)

End-to-end eval for Citeflow retrieval and graph on HotpotQA **distractor** validation.

Benchmark knobs are **CLI only** (no `benchmarking/hotpotqa/.env`). Secrets stay in **repo root `.env`**.

## Prerequisites

- Python env with repo dependencies (e.g. `rag_env_1`)
- Repo root `.env`: OpenAI, Qdrant, Jina, `USE_BM25`, `USE_LATE_INTERACTION`, retrieval limits, graph models

Run all commands from the **repo root**.

## Steps

```bash
# 1. Download & process (distractor validation)
python -m benchmarking.hotpotqa.download_process_hotpotqa --max-questions 50

# 2. Upload: one command can create several PQ collections (embeddings computed once)
python -m benchmarking.hotpotqa.upload_qdrant_embedding \
  --no-enrich \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32

# 3. Evaluate sequentially (none, then pq8, …). Each PQ writes its own folder.
python -m benchmarking.hotpotqa.run_evaluation \
  --mode retrieval \
  --strategy fast_bm25_retrieval \
  --experiment-name hotpot_retrieval_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32

python -m benchmarking.hotpotqa.run_evaluation \
  --mode graph \
  --experiment-name hotpot_graph_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32
```

`--collection-base` must match upload. Collections are `{base}_none`, `{base}_pq8`, `{base}_pq16`, `{base}_pq32`.

`--experiment-name` is required. Results go to `data/results/<experiment-name>/<pq>/`. If that PQ folder already exists, the run exits without overwriting.

`--max-questions` on eval (default `0`) slices the processed JSON; `0` uses the full downloaded file.

`--ragas-model` defaults to `gpt-4o-mini` (context recall only).

The graph is invoked as-is; eval only switches `QDRANT_COLLECTION_NAME` / `DENSE_PQ` on the shared settings object.

## Outputs

Under `benchmarking/hotpotqa/data/results/<experiment-name>/<pq>/`:

- `results.json` — per-question eval rows
- `run_metadata.json` — mode, PQ, collection, latency, env snapshot
- `ragas_results.json` — context recall scores
- `benchmark_report.md` — context recall + latencies

## Metrics

| Metric | Retrieval | Graph |
|--------|-----------|-------|
| Context recall | yes | yes |
| Latency | retriever wall time | full graph wall time |

## Environment

| Variable | File |
|----------|------|
| API keys, Qdrant URL, BM25 / late-interaction / MMR / top-k | repo root `.env` |
| Max questions, PQ list, collection base, experiment name, RAGAS model | CLI |
