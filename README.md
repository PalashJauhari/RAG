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
  `exploratory_query`, plus an initial **`retrieval_strategy`** tier for the retriever.
- **Specialized Query Preparation**:
  - Simple queries go directly to retrieval.
  - Comparison, multihop, and procedural queries are split into focused retrieval queries.
  - Exploratory queries are expanded into multiple retrieval angles.
  - Ambiguous queries are rewritten safely for now; the `/resume` endpoint remains available for
    future clarification support.
- **Evidence Evaluation Loop**: The evaluator returns `sufficient`, `insufficient_recall`,
  `intent_mismatch`, or **`strategy_upgrade`** (queries OK but retrieval tier too weak—rerun retrieval
  with a heavier strategy without gap-fill or intent rewrite). Recall gaps route through gap-fill;
  intent mismatches route through intent-correction. Gap-fill and intent nodes may optionally bump
  **`retrieval_strategy`**. If retry budgets are exhausted, the graph routes to `partial_answer`.
  Structured outputs are validated in `output_validation/` (see **Output validation** below).
- **Per-request retrieval strategies** (chosen by complexity / evaluator / gap / intent):
  - **`fast_retrieval`**: dense (+ optional MMR per deployment settings).
  - **`fast_bm25_retrieval`**: dense + BM25 + RRF fusion.
  - **`keyword`**: BM25-only (no dense embeddings for that pass).
  - **`fast_bm25_late_interaction_retrieval`**: hybrid fusion then ColBERT-style late interaction when Jina is configured.
- **Structured turn trace**: Append-only **`message_query`** audit rows across nodes; a terminal
  **`clear_turn_trace`** node resets the trace after each answer using LangGraph **`Overwrite([])`**
  so multi-turn threads do not leak prior-turn diagnostics into the next query.
- **Lean Message State**: `messages` stores user turns and final answer node outputs only. Node
  scratch data lives in explicit keys such as `normalized_query`, `active_retrieval_queries`,
  `retrieval_strategy`, `message_query`, `retrieved_documents`, and `information_evaluation`.
- **Strict Grounding**: The final answer node answers only from retrieved documents. Source citation
  wiring is intentionally deferred, so `sources` is currently returned as an empty array. The final
  answer prompt receives the normalized query, retrieval strategy, active retrieval queries, and
  retrieved documents. Partial answer also receives `information_evaluation`. Mid-graph nodes
  (evaluator, gap-fill, intent) still receive the slim `message_query` trace for retry history.
- **Slim audit trace**: `message_query` rows avoid duplicating live state (e.g. no `prep` — use
  `node`; evaluator rows carry `evaluation_status` only; retrieval rows carry `queries`, `strategy`,
  `new_doc_count`). Gap details live in `information_evaluation`, not a separate state key.

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
        -> answer_node -> clear_turn_trace_node -> END
      insufficient_recall with retries left
        -> gap_fill_node
        -> retrieval_node
      intent_mismatch with retries left
        -> intent_correction_rewriter_node
        -> retrieval_node
      strategy_upgrade with retries left
        -> retrieval_node  # same active_retrieval_queries; heavier retrieval_strategy
      insufficient_recall / intent_mismatch / strategy_upgrade exhausted
        -> partial_answer_node -> clear_turn_trace_node -> END
```

The graph appends retrieved documents across retry loops within the same user turn. Each `/run`
input resets turn-local scratch such as `active_retrieval_queries`, `retrieval_strategy`,
`retrieved_documents`, and evaluator retry counters. The append-only **`message_query`** trace is
cleared **after** `answer` / `partial_answer` by **`clear_turn_trace_node`** using LangGraph
**`Overwrite([])`** (turn-local invokes still pass `message_query: []` with other scratch, but the
authoritative reset for reducer-backed history is the terminal clear node).

`/resume` is still exposed by the API so clarification can be reintroduced later without changing
the client contract. The current graph does not interrupt for ambiguous queries; it rewrites them
best-effort and continues to retrieval.

## Advanced Retrieval Flow

The `Retriever.retrieve(queries, strategy)` branch selects behavior **per request**:

| Strategy | Dense (+MMR if enabled in settings) | BM25 | RRF fusion | Late interaction |
|----------|-------------------------------------|------|------------|------------------|
| `fast_retrieval` | yes | no | no | no |
| `fast_bm25_retrieval` | yes | yes | yes | no |
| `keyword` | no | yes | no | no |
| `fast_bm25_late_interaction_retrieval` | yes | yes | yes | yes |

Environment flags **`USE_BM25`**, **`USE_LATE_INTERACTION`**, and **`USE_MMR`** still configure the
Qdrant client (e.g. cloud inference), embedding/MMR parameters, and Jina availability; the
**`strategy`** argument chooses which branches run inside `_retrieve_one`.

For multiple active retrieval queries, the retriever runs each query, deduplicates by point id, and boosts
documents that rank well across query result sets (RRF across queries).

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
STRATEGY_UPGRADE_MAX_RETRIES=3

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

### Stream Graph Progress

The existing `/run` endpoint remains unchanged. For node-level progress streaming, use:

```bash
curl -N -X POST http://127.0.0.1:8000/run/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-stream","message":"Compare the refund policies for Enterprise and Consumer tiers."}'
```

`/run/stream` returns Server-Sent Events (`text/event-stream`). Each event is a JSON object in a
`data:` frame. The stream reports graph progress after each node completes; it does not stream final
answer tokens word by word.

Example events:

```text
data: {"type":"node","node":"query_normalisation","status":"completed","label":"Normalizing query","normalized_query":"Compare the Enterprise refund policy with the Consumer refund policy."}

data: {"type":"node","node":"query_complexity","status":"completed","label":"Classifying query complexity","complexity":"comparison_query","retrieval_strategy":"fast_bm25_retrieval","explanation":"..."}

data: {"type":"node","node":"query_splitter","status":"completed","label":"Preparing retrieval queries","active_retrieval_queries":["Enterprise refund policy","Consumer refund policy"]}

data: {"type":"node","node":"retrieval","status":"completed","label":"Retrieving documents","retrieval_strategy":"fast_bm25_retrieval","retrieval_queries":["Enterprise refund policy","Consumer refund policy"],"new_retrieved_documents":[{"score":0.42,"text":"..."}],"retrieved_doc_count":8}

data: {"type":"node","node":"information_evaluator","status":"completed","label":"Evaluating evidence","evaluation_status":"sufficient","next_retrieval_strategy":null,"missing_evidence_details":[],"insufficient_recall_retry_count":0,"intent_mismatch_retry_count":0,"strategy_upgrade_retry_count":0}

data: {"type":"node","node":"information_evaluator","status":"completed","label":"Evaluating evidence","evaluation_status":"insufficient_recall","next_retrieval_strategy":null,"missing_evidence_details":["Passages cover Enterprise refunds but not Consumer tier deadlines."],"insufficient_recall_retry_count":1,"intent_mismatch_retry_count":0,"strategy_upgrade_retry_count":0}

data: {"type":"final","node":"answer","status":"completed","label":"Answer ready","answer":"...","sources":[],"confidence":"high"}

data: {"type":"node","node":"clear_turn_trace","status":"completed","label":"Clearing turn trace"}

data: {"type":"done","session_id":"demo-stream","retrieved_docs":[{"score":0.42,"text":"..."}]}
```

The closing `done` frame includes `retrieved_docs` (compact `score` / `text` rows, matching `/run`)
so streaming clients can show passage counts or tooling without a second `/run` invoke.

If retry budgets are exhausted, the final event comes from `partial_answer` instead of `answer`.
Retrieval `node` frames include `new_retrieved_documents` and `retrieved_doc_count` for **this
retrieval pass only** (not the full accumulated corpus). The terminal `done` frame carries all
compact docs accumulated across retry loops in the turn.

## Output validation

Evaluator and retrieval-tier rules are enforced in code and at parse time:

### Retrieval tier order

Canonical low → high order (`RETRIEVAL_STRATEGY_ORDER` in `output_validation/retrieval_strategy.py`):

```text
fast_retrieval → keyword → fast_bm25_retrieval → fast_bm25_late_interaction_retrieval
```

`strategy_upgrade` must request a tier **strictly above** the current `retrieval_strategy`. After the
LLM responds, `resolve_strategy_upgrade()` in `output_validation/information_evaluator.py` clamps
invalid or equal/lighter choices to the minimum heavier tier when one exists.

### `insufficient_recall`

When `evaluation_status` is `insufficient_recall`, `missing_evidence_details` must contain at least
one non-empty string (Pydantic on `InformationEvaluation`). Other statuses must leave this array empty.

### `strategy_upgrade` at max tier

If the current tier is already `fast_bm25_late_interaction_retrieval` and the evaluator still requests
`strategy_upgrade`, the graph sets `strategy_upgrade_retry_count` to the configured maximum so the
same turn routes to **`partial_answer`** without another retrieval pass.

## Development Notes

- **Prompts**: Centralized in `prompts/`, one system prompt per graph node.
- **Output Validation**: Structured node outputs live in `output_validation/` and use Pydantic
  models with descriptive fields (evaluator rules above).
- **LLM Client**: Unified client construction in `middleware/llm_client.py` handles model
  configuration, structured output wrappers, and rate limiting.
- **Metadata Handling**: Final answer messages preserve provider metadata in `messages`. Prompt
  contexts are built as plain text or JSON so LangChain response metadata is not sent back to LLMs.
- **Observability**: Langfuse integration is available at API, graph node, summarization, and
  LangChain callback layers when `LANGFUSE_TRACING_ENABLED=true`.
- **UI**: The Dash app in `ui/dash_app.py` mirrors the **ForecastingPlatform** viewport shell (muted
  outer `#DCDCD8`, framed `#FAFAF8` card, left rail, chat column, rounded composer). It calls
  **`POST /run/stream`** from the browser (async fetch + SSE) and appends each graph step to a
  tall **Progress** column on the **right** (`strong` = LangGraph **node id**, remainder = detail).
  Steps include **`clear_turn_trace`** after final or partial answers. The final answer appears in
  the chat bubbles; **`POST /resume`** (JSON) is still used when the API reports an interrupt. Set **`API_URL`** via environment / `dcc.Store` defaults so the UI reaches
  the same FastAPI origin as `uvicorn`. For Dash on **8050**, the API enables **CORS** for
  loopback origins (`127.0.0.1`, `localhost`, `0.0.0.0`, `[::1]`) plus a small localhost regex so
  OPTIONS preflight from the UI succeeds. Styles live in `ui/assets/rag_styles.css`;
  streaming helpers in `ui/assets/rag_ui.js`. Programmatic consumers can use
  `RagApiClient.iter_run_stream` in `ui/api_client.py`.

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
