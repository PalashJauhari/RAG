# HotpotQA benchmark (distractor)

End-to-end eval for Factline retrieval and graph on HotpotQA **distractor** validation.

## Prerequisites

- Python env with repo dependencies (e.g. `rag_env_1`)
- Repo root `.env`: OpenAI, Qdrant, Jina, `USE_BM25`, `USE_LATE_INTERACTION`, retrieval limits, graph models, `QDRANT_COLLECTION_NAME`
- `benchmarking/hotpotqa/.env`: only three variables (see `.env.example`)

Run all commands from the **repo root**.

## Steps

```bash
# 1. Download & process (distractor validation)
python -m benchmarking.hotpotqa.download_process_hotpotqa --max-questions 200

# 2. Upload to Qdrant (pick one)
python -m benchmarking.hotpotqa.upload_qdrant_embedding --no-enrich
python -m benchmarking.hotpotqa.upload_qdrant_embedding --enrich

# 3. Evaluate (pick one)
python -m benchmarking.hotpotqa.run_evaluation --mode retrieval --strategy fast_bm25_retrieval
python -m benchmarking.hotpotqa.run_evaluation --mode graph
```

Optional: `--experiment-name NAME` on eval overrides `HOTPOTQA_EXPERIMENT_NAME`.

Align `QDRANT_COLLECTION_NAME` in root `.env` with the collection you upload to.

## Outputs

Under `benchmarking/hotpotqa/data/results/{HOTPOTQA_EXPERIMENT_NAME}/`:

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
| `HOTPOTQA_EXPERIMENT_NAME` | `benchmarking/hotpotqa/.env` |
| `HOTPOTQA_RAGAS_MODEL` | `benchmarking/hotpotqa/.env` |
| Everything else (API keys, Qdrant, models, flags) | repo root `.env` |

Enrichment uses `QUERY_DECOMPOSITION_MODEL` from root `.env` when `--enrich` is set.
