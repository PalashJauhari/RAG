# Advanced RAG Orchestration Pipeline

This project implements a production-grade Retrieval-Augmented Generation (RAG) pipeline built
on **LangGraph**, **Qdrant**, and **OpenAI**.

The system uses an explicit LangGraph flow that normalizes the user's query, classifies query
complexity, prepares retrieval queries, retrieves evidence, evaluates whether the evidence matches
the intent, repairs recall or intent issues when needed, and only then produces a grounded answer.

## Key Features

- **Query Normalisation First**: The latest user message is rewritten into a standalone query using
  conversation context while preserving ambiguity instead of guessing.
- **Typed Complexity Routing**: Query complexity is emitted as validated JSON with one exact label:
  `simple_query`, `comparison_query`, `multihop_query`, `procedural_query`, `ambiguous_query`, or
  `exploratory_query`.
- **Specialized Query Preparation**:
  - Simple queries go directly to retrieval.
  - Comparison, multihop, and procedural queries are split into focused retrieval queries.
  - Exploratory queries are expanded into multiple retrieval angles.
  - Ambiguous queries are rewritten safely for now; the `/resume` endpoint remains available for
    future clarification support.
- **Evidence Evaluation Loop**: The evaluator returns `sufficient`, `insufficient_recall`, or
  `intent_mismatch`. Recall gaps route through gap-fill query generation; intent mismatches route
  through intent-correction rewriting. If retry budgets are exhausted, the graph routes to a
  partial-answer node instead of pretending the evidence is complete.
- **3-Stage Hybrid Retrieval**:
  1. **Stage 1 (Base Retrieval)**: Concurrent Dense (with optional MMR over-fetch) and Sparse
     BM25 searches.
  2. **Stage 2 (Fusion)**: Reciprocal Rank Fusion (RRF) to merge dense and sparse candidates.
  3. **Stage 3 (Re-ranking)**: Optional ColBERTv2 late-interaction re-ranking.
- **Lean Message State**: `messages` stores user turns and final answer node outputs only. Node
  scratch data lives in explicit state keys such as `normalized_query`, `parsed_queries`,
  `retrieved_documents`, and `information_evaluation`.
- **Strict Grounding**: The final answer node answers only from retrieved documents. Source citation
  wiring is intentionally deferred, so `sources` is currently returned as an empty array. Final and
  partial answer prompts receive the normalized query, parsed query lists, and retrieved documents,
  but only retrieved passages are treated as evidence.

## Architecture

```text
FastAPI /run
  -> query_normalisation_node
  -> query_complexity_node
      simple_query
        -> retrieval_node
      comparison_query / multihop_query / procedural_query
        -> query_splitter_node
        -> retrieval_node
      exploratory_query
        -> query_expansion_node
        -> retrieval_node
      ambiguous_query
        -> query_rewriter_node
        -> retrieval_node
  -> information_evaluator_node
      sufficient
        -> answer_node
      insufficient_recall with retries left
        -> gap_fill_node
        -> retrieval_node
      intent_mismatch with retries left
        -> intent_correction_rewriter_node
        -> retrieval_node
      insufficient_recall / intent_mismatch exhausted
        -> partial_answer_node
```

The graph appends retrieved documents across retry loops within the same user turn. A new `/run`
input resets turn-level scratch fields such as `parsed_queries`, `retrieved_documents`, evaluator
state, and retry counters. Gap-fill queries are stored in `parsed_queries_insufficient_recall`;
intent-correction queries are stored separately in `parsed_queries_intent_correction` so the
original parsed queries remain available for debugging and final answer context.

`/resume` is still exposed by the API so clarification can be reintroduced later without changing
the client contract. The current graph does not interrupt for ambiguous queries; it rewrites them
best-effort and continues to retrieval.

## Advanced Retrieval Flow

The `Retriever` executes a multi-stage pipeline for every query:

1. **Candidate Fetching**
   - **Dense**: Fetches candidates using OpenAI embeddings. With MMR enabled, Qdrant fetches a
     larger internal candidate pool to encourage diversity.
   - **Sparse (BM25)**: Optional keyword search using Qdrant cloud inference.
2. **Hybrid Fusion**
   - Uses **Reciprocal Rank Fusion (RRF)** to merge dense and sparse pools.
3. **Late Interaction Reranking**
   - Uses **ColBERTv2** via Jina multi-vectors to rerank the fused candidate pool.
   - Returns the final `RETRIEVAL_TOP_K` documents.

For multiple parsed queries, the retriever runs each query, deduplicates by point id, and boosts
documents that rank well across query result sets.

## Environment Configuration

Copy `.env.example` to `.env`. Key flags:

```env
# Retrieval Strategy
USE_BM25=true
USE_LATE_INTERACTION=true
USE_MMR=true
RETRIEVAL_CANDIDATE_LIMIT=100
RETRIEVAL_TOP_K=8

# Context Management
MESSAGE_SUMMARY_TOKEN_THRESHOLD=100000
MESSAGE_SUMMARY_KEEP_RECENT=10

# Retry Loops
INSUFFICIENT_RECALL_MAX_RETRIES=3
INTENT_MISMATCH_MAX_RETRIES=2

# Node Models
QUERY_NORMALISATION_MODEL=gpt-4.1-mini
QUERY_COMPLEXITY_MODEL=gpt-4.1-mini
QUERY_REWRITER_MODEL=gpt-4.1-mini
INFORMATION_EVALUATOR_MODEL=gpt-4.1-mini
FINAL_ANSWER_MODEL=gpt-4.1-mini
QUERY_DECOMPOSITION_MODEL=gpt-4.1-mini
QUERY_EXPANSION_MODEL=gpt-4.1-mini
GAP_FILL_MODEL=gpt-4.1-mini
INTENT_CORRECTION_REWRITER_MODEL=gpt-4.1-mini

# Observability
LANGFUSE_TRACING_ENABLED=false
```

## API Usage

Start the server:

```bash
uvicorn api.main:app --reload
```

### Run a query

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Compare the refund policies for Enterprise and Consumer tiers."}'
```

### Response Structure

The system returns a structured JSON response:

- `answer`: Grounded response based strictly on retrieved documents.
- `sources`: Empty for now; source extraction will be wired later.
- `confidence`: `high`, `medium`, or `low`.
- `retrieved_docs`: Last turn's compact retrieved documents, currently `score` and `text`.

## Development Notes

- **Prompts**: Centralized in `prompts/`, one system prompt per graph node.
- **Output Validation**: Structured node outputs live in `output_validation/` and use Pydantic
  models with descriptive fields.
- **LLM Client**: Unified client construction in `middleware/llm_client.py` handles model
  configuration, structured output wrappers, and rate limiting.
- **Metadata Handling**: Final answer messages preserve provider metadata in `messages`. Prompt
  contexts are built as plain text or JSON so LangChain response metadata is not sent back to LLMs.
- **Observability**: Langfuse integration is available at API, graph node, summarization, and
  LangChain callback layers when `LANGFUSE_TRACING_ENABLED=true`.
- **UI**: The Dash app in `ui/dash_app.py` talks to `/run` and `/resume` through `ui/api_client.py`.

### Plot the LangGraph Flow

Render the compiled graph to Mermaid source and a PNG image:

```bash
python scripts/plot_langgraph.py
```

Defaults:

```text
artifacts/langgraph.mmd
artifacts/langgraph.png
```

The default PNG renderer uses Mermaid.ink through LangChain Core's `draw_mermaid_png()` API path.
Use `--draw-method pyppeteer` if you have Pyppeteer installed and want local rendering.

## Benchmarking

HotpotQA validation and RAGAS metrics live in `benchmarking/hotpotqa`. This suite evaluates the
retriever directly; it does not run the full LangGraph agent.

```bash
python -m benchmarking.hotpotqa.dataset.prepare_eval_data
python -m benchmarking.hotpotqa.qdrant_upload.upload
python -m benchmarking.hotpotqa.evaluation.run_retrieval_eval
python -m benchmarking.hotpotqa.metrics.ragas_metrics
python -m benchmarking.hotpotqa.metrics.exact_metrics
```

---

Powered by LangGraph, Qdrant Cloud, and Advanced RAG Research.
