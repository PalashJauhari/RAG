# HotpotQA evaluation

This folder runs **offline Citeflow eval** on [HotpotQA](https://hotpotqa.github.io/) (distractor validation). You index the supporting (and distractor) paragraphs, ask the same questions Citeflow would see in production, then score the answers.

HotpotQA is a multi-hop set: each question usually needs **more than one paragraph**. That stresses recall repair and citation, not just “did the top hit look relevant?”

Typical use: compare **search only** vs the **full Citeflow graph**, and optionally compare dense product-quantization (PQ) settings on the same embeddings.

Knobs are **CLI flags**. Keys (OpenAI, Qdrant, Jina) stay in the **repo root `.env`**. There is no `benchmarking/hotpotqa/.env`.

Run every command from the **repo root**.

---

## What you get

- A processed question file (ids, gold answers, supporting vs distractor contexts)
- One or more Qdrant collections: `{collection-base}_{none|pq8|pq16|pq32}`
- Per-question JSON plus a short markdown report you can paste into the [root README](../../README.md) table
- RAGAS scores: **context recall** always; **faithfulness** and **answer correctness** on graph runs
- **Partial-answer** rate on graph runs (when Citeflow refuses to guess a full answer)
- **Mean and p50 latency in seconds** (retriever wall time, or full graph wall time)

The graph itself is not forked for eval. The runner only points `QDRANT_COLLECTION_NAME` and `DENSE_PQ` at the collection under test.

---

## How a run works

```mermaid
flowchart TD
  download[Download HotpotQA] --> upload[Embed once and upsert]
  upload --> retrieve[Retrieval eval]
  upload --> graph[Graph eval]
  retrieve --> reportR[Recall plus latency]
  graph --> reportG[Recall, faithfulness, correctness, partials, latency]
```

1. **Download** the distractor split and write a local JSON (cap with `--max-questions`).
2. **Upload** contexts to Qdrant. Embeddings are computed once; you can create several PQ collections in the same command.
3. **Evaluate** one mode at a time. `--pq` is sequential (none, then pq8, …). Each PQ writes its own folder.
4. **Score** with RAGAS, then write `benchmark_report.md`.

Retrieval mode answers “did search surface the right passages?” Graph mode answers “did the whole Citeflow turn stay grounded and correct?”

---

## Quick start

Root `.env` must already have OpenAI, Qdrant, and (for ColBERT rerank) Jina, plus `USE_BM25` / `USE_LATE_INTERACTION` matching how you want to search.

```bash
# 1. Download & process (default cap is 200; 0 = full split)
python -m benchmarking.hotpotqa.download_process_hotpotqa --max-questions 50

# 2. Upload: several PQ collections, embeddings computed once
python -m benchmarking.hotpotqa.upload_qdrant_embedding \
  --no-enrich \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32

# 3a. Search only (hybrid). Repeat with fast_bm25_late_interaction_retrieval for rerank.
python -m benchmarking.hotpotqa.run_evaluation \
  --mode retrieval \
  --strategy fast_bm25_retrieval \
  --experiment-name hotpot_retrieval_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32

# 3b. Full Citeflow pipeline
python -m benchmarking.hotpotqa.run_evaluation \
  --mode graph \
  --experiment-name hotpot_graph_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32
```

`--collection-base` must match upload. Collections are `{base}_none`, `{base}_pq8`, `{base}_pq16`, `{base}_pq32`.

`--experiment-name` is required. Results go to `benchmarking/hotpotqa/data/results/<experiment-name>/<pq>/`. If that folder already exists, the run **exits** instead of overwriting.

`--max-questions` on eval (default `0`) slices the processed JSON; `0` uses the whole downloaded file.

`--ragas-model` defaults to `gpt-4o-mini`.

Retrieval `--strategy` values: `fast_retrieval`, `keyword`, `fast_bm25_retrieval` (hybrid), `fast_bm25_late_interaction_retrieval` (hybrid + ColBERT).

---

## Metrics

Retrieval-only numbers on this page and the [root README](../../README.md) (distractor split, **200** questions). **Context recall** is RAGAS. **Latency** is mean retriever wall time in seconds. Graph mode still writes faithfulness, answer correctness, and partial-answer rate into the per-run report; those are not in the tables below yet.

### Context recall

**Dense + BM25**

| Setup | No quantization | PQ-8 | PQ-16 | PQ-32 |
|-------|-----------------|------|-------|-------|
| Without enrichment | 0.81 | 0.80 | 0.81 | 0.82 |
| With enrichment | 0.81 | 0.80 | 0.82 | 0.81 |

**Dense + BM25 + ColBERT rerank**

| Setup | No quantization | PQ-8 | PQ-16 | PQ-32 |
|-------|-----------------|------|-------|-------|
| Without enrichment | 0.87 | 0.86 | 0.86 | 0.86 |
| With enrichment | 0.85 | 0.86 | 0.85 | 0.86 |

### Mean latency (seconds)

**Dense + BM25**

| Setup | No quantization | PQ-8 | PQ-16 | PQ-32 |
|-------|-----------------|------|-------|-------|
| Without enrichment | 0.54 | 0.59 | 0.53 | 0.60 |
| With enrichment | 0.52 | 0.53 | 0.55 | 0.57 |

**Dense + BM25 + ColBERT rerank**

| Setup | No quantization | PQ-8 | PQ-16 | PQ-32 |
|-------|-----------------|------|-------|-------|
| Without enrichment | 11.58 | 12.18 | 12.77 | 12.93 |
| With enrichment | 12.09 | 10.50 | 11.21 | 13.23 |

---

## Outputs

Under `benchmarking/hotpotqa/data/results/<experiment-name>/<pq>/`:

| File | What it is |
|------|------------|
| `results.json` | Per-question rows (answers, contexts, flags, latency) |
| `run_metadata.json` | Mode, PQ, collection, latency summary, env snapshot |
| `ragas_results.json` | Per-question RAGAS scores |
| `benchmark_report.md` | Table you can copy from |

---

## Configuration

| What | Where |
|------|--------|
| API keys, Qdrant URL, BM25 / late-interaction / MMR / top-k, graph models | repo root `.env` |
| Question cap, PQ list, collection base, experiment name, RAGAS model, retrieval strategy | CLI |

Do not change `graph/graph.py` to run this eval.
