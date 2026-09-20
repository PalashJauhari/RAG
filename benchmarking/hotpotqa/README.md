# HotpotQA benchmark

Reproducible retrieval and end-to-end evaluation for Citeflow using the [HotpotQA](https://hotpotqa.github.io/) distractor validation split.

HotpotQA is a multi-hop benchmark: answering a question often requires evidence from more than one paragraph. This makes it useful for testing retrieval coverage, reranking, repair loops, grounding, and answer quality.

All commands run from the repository root. API keys and retrieval settings come from the root `.env`; this folder has no separate environment file.

## Results

The retrieval benchmark uses **200 questions**, returns five passages per question, and evaluates every result with RAGAS. Values below are means across all 200 questions and are rounded to two decimals.

| Retrieval | Quantization | Context recall | Context precision | Mean latency |
|---|---:|---:|---:|---:|
| Dense + BM25 | None | 0.82 | 0.48 | 0.51 s |
| Dense + BM25 | PQ-32 | 0.81 | 0.50 | 0.53 s |
| **Dense + BM25** | **PQ-16** | **0.80** | **0.49** | **0.52 s** |
| Dense + BM25 | PQ-8 | 0.80 | 0.48 | 0.52 s |
| Dense + BM25 + ColBERT | None | 0.86 | 0.54 | 4.90 s |
| Dense + BM25 + ColBERT | PQ-32 | 0.87 | 0.52 | 5.22 s |
| **Dense + BM25 + ColBERT** | **PQ-16** | **0.87** | **0.53** | **4.80 s** |
| Dense + BM25 + ColBERT | PQ-8 | 0.86 | 0.52 | 5.20 s |

Context recall measures how much of the gold answer is supported by retrieved passages. Context precision measures whether useful passages are ranked ahead of irrelevant ones. Latency is retriever wall time and does not include RAGAS scoring.

| Workflow | Quantization | Context recall | Context precision | Faithfulness | Factual correctness | Relevancy | Partial answers | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Citeflow graph** | **PQ-16** | **0.89** | **0.59** | **0.86** | **0.31** | **0.48** | **24.50%** | **28.22 s** |

## Evaluation modes

### Retrieval

Compares search strategies without running the Citeflow graph:

- Context recall
- Context precision
- Retriever latency

Supported strategies include dense retrieval, keyword search, Dense + BM25, and Dense + BM25 with ColBERT reranking.

### Graph

Runs the complete Citeflow workflow and reports:

- Context recall and context precision
- Faithfulness
- Factual correctness
- Response relevancy (`N=3`)
- Partial-answer rate
- End-to-end graph latency

## Run the benchmark

### 1. Download and process HotpotQA

```bash
python -m benchmarking.hotpotqa.download_process_hotpotqa \
  --max-questions 200
```

### 2. Build Qdrant collections

Embeddings are computed once and reused across all product-quantization settings.

```bash
python -m benchmarking.hotpotqa.upload_qdrant_embedding \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32
```

### 3. Run Dense + BM25

```bash
python -m benchmarking.hotpotqa.run_evaluation \
  --mode retrieval \
  --strategy fast_bm25_retrieval \
  --experiment-name hotpot_retrieval_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32
```

### 4. Run Dense + BM25 with ColBERT

```bash
python -m benchmarking.hotpotqa.run_evaluation \
  --mode retrieval \
  --strategy fast_bm25_late_interaction_retrieval \
  --experiment-name hotpot_retrieval_rerank_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32
```

### 5. Run the full graph

```bash
python -m benchmarking.hotpotqa.run_evaluation \
  --mode graph \
  --experiment-name hotpot_graph_v1 \
  --collection-base hotpotqa_eval \
  --pq none pq8 pq16 pq32
```

`--collection-base` must match the upload command. Experiment names must be unique: existing result folders are never overwritten. Add `--max-questions N` to evaluate only the first `N` processed questions.

## Output

Each configuration writes to:

```text
benchmarking/hotpotqa/data/results/<experiment-name>/<pq>/
```

| File | Contents |
|---|---|
| `results.json` | Questions, retrieved contexts, answers, flags, and per-question latency |
| `run_metadata.json` | Run configuration and aggregate latency |
| `ragas_results.json` | Per-question RAGAS scores and metric means |
| `benchmark_report.md` | Human-readable summary and type/level breakdown |

## Reproducibility

The evaluated retrieval setup used:

- Final results: 5
- Dense candidates: 50 with MMR (`0.5` diversity)
- BM25 candidates: 50
- ColBERT reranking pool: 25
- PQ variants: none, PQ-8, PQ-16, PQ-32
- RAGAS model: `gpt-4o-mini`

Runtime settings are also captured in every `run_metadata.json`.
