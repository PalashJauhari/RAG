# HotpotQA benchmark (distractor)

End-to-end eval for Factline retrieval and graph on HotpotQA **distractor** validation.

## Prerequisites

- Python env with repo dependencies (e.g. `rag_env_1`)
- Repo root `.env`: OpenAI, Qdrant, Jina, `USE_BM25`, `USE_LATE_INTERACTION`, retrieval limits, graph models, `QDRANT_COLLECTION_NAME`
- `benchmarking/hotpotqa/.env`: `HOTPOTQA_MAX_QUESTIONS`, `HOTPOTQA_RAGAS_MODEL` (see `.env.example`)

Run all commands from the **repo root**.

## Steps

```bash
# 1. Download & process (distractor validation)
python -m benchmarking.hotpotqa.download_process_hotpotqa --max-questions 200

# 2. Upload to Qdrant (pick one)
python -m benchmarking.hotpotqa.upload_qdrant_embedding --no-enrich
python -m benchmarking.hotpotqa.upload_qdrant_embedding --enrich

# 3. Evaluate (required: unique --experiment-name per run)
python -m benchmarking.hotpotqa.run_evaluation \
  --mode retrieval \
  --strategy fast_bm25_retrieval \
  --experiment-name hotpot_retrieval_bm25_v1

python -m benchmarking.hotpotqa.run_evaluation \
  --mode graph \
  --experiment-name hotpot_graph_v1
```

`--experiment-name` is **required** for both modes. It is not read from `.env`. If `data/results/<name>/` already exists, the run exits without overwriting.

Align `QDRANT_COLLECTION_NAME` in root `.env` with the collection you upload to.

## Outputs

Under `benchmarking/hotpotqa/data/results/<experiment-name>/`:

- `results.json` — per-question eval rows
- `run_metadata.json` — mode, latency, env snapshot
- `ragas_results.json` — RAGAS scores
- `benchmark_report.md` — summary report

## Metrics

| Metric | Retrieval | Graph |
|--------|-----------|-------|
| Context precision / recall | yes | yes |
| Faithfulness | — | yes |
| Answer correctness | — | yes |

Partial graph answers are included in RAGAS when scorable.

## Environment

| Variable | File |
|----------|------|
| `HOTPOTQA_MAX_QUESTIONS` | `benchmarking/hotpotqa/.env` |
| `HOTPOTQA_RAGAS_MODEL` | `benchmarking/hotpotqa/.env` |
| Everything else (API keys, Qdrant, models, flags) | repo root `.env` |

Experiment name: **`--experiment-name` CLI only** (eval step).

Enrichment uses `QUERY_DECOMPOSITION_MODEL` from root `.env` when `--enrich` is set.
