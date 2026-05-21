# Advanced RAG Orchestration Pipeline

This project implements a production-grade Retrieval-Augmented Generation (RAG) pipeline built
on **LangGraph**, **Qdrant**, and **OpenAI**.

The system uses an explicit LangGraph flow that normalizes the user's query, classifies query
complexity, prepares retrieval queries, retrieves evidence, runs fact-based recall check and intent
alignment, repairs gaps when needed, and only then produces a grounded answer.

## Key Features

- **Query Normalisation First**: The latest user message is rewritten into a standalone query using
  conversation context while preserving ambiguity instead of guessing.
- **Typed Complexity Routing**: Query complexity is emitted as validated JSON with one exact label:
  `simple_query`, `comparison_query`, `multihop_query`, `procedural_query`, `ambiguous_query`, or
  `exploratory_query`. Retrieval tier selection is deterministic and not predicted by this node.
- **Specialized Query Preparation**:
  - Simple queries go directly to retrieval.
  - Comparison, multihop, and procedural queries are split into focused retrieval queries.
  - Exploratory queries are expanded into multiple retrieval angles.
  - Ambiguous queries are rewritten safely for now; the `/resume` endpoint remains available for
    future clarification support.
- **Recall and repair loop**: `recall_check` decomposes the normalized query into `fact1`, `fact2`,
  ... information needs, then verifies each fact against retrieved passages with evidence excerpts.
  If any fact is unsupported, `intent_check` routes per fact to gap-fill or intent-correction.
  A single **`retrieval_retry_count`** caps loops. Structured outputs are validated in
  `output_validation/`.
- **Per-request retrieval strategies** (chosen deterministically by retry count):
  - **`fast_retrieval`**: dense (+ optional MMR per deployment settings).
  - **`fast_bm25_retrieval`**: dense + BM25 + RRF fusion.
  - **`keyword`**: BM25-only (no dense embeddings for that pass).
  - **`fast_bm25_late_interaction_retrieval`**: hybrid fusion then ColBERT-style late interaction when Jina is configured.
  The graph uses `fast_bm25_retrieval` by default and switches to
  `fast_bm25_late_interaction_retrieval` when `retrieval_retry_count >= max(0, RETRIEVAL_LOOP_MAX_RETRIES - 2)`.
- **Structured turn trace**: Append-only **`message_query`** audit rows across nodes; a terminal
  **`clear_turn_trace`** node resets the trace after each answer using LangGraph **`Overwrite([])`**
  so multi-turn threads do not leak prior-turn diagnostics into the next query.
- **Lean Message State**: `messages` stores user turns and final answer node outputs only. Node
  scratch data lives in explicit keys such as `normalized_query`, `active_retrieval_queries`,
  `retrieval_strategy`, `message_query`, `retrieved_documents`, `required_facts`,
  `fact_verifications`, and `unsupported_fact_keys`.
- **Strict Grounding**: The final answer node answers only from retrieved documents. Source citation
  wiring is intentionally deferred, so `sources` is currently returned as an empty array. The final
  answer prompt receives the normalized query, retrieval strategy, active retrieval queries, and
  retrieved documents. Partial answer also receives `fact_verifications` and retry context.
- **Slim audit trace**: `message_query` rows avoid duplicating live state; retrieval rows carry
  `queries`, `strategy`, `new_doc_count`.

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
  -> recall_check_node
      recall sufficient -> answer_node -> clear_turn_trace_node -> END
      retries exhausted -> partial_answer_node -> clear_turn_trace_node -> END
      recall insufficient -> intent_check_node
          intent ok -> gap_fill_node -> strategy_upgrade_node -> retrieval_node
          intent wrong -> intent_correction_rewriter_node -> strategy_upgrade_node -> retrieval_node
```

The graph appends retrieved documents across retry loops within the same user turn. Each `/run`
input resets turn-local scratch such as `active_retrieval_queries`, `retrieval_strategy`,
`retrieved_documents`, `required_facts`, `fact_verifications`, `unsupported_fact_keys`, and
`retrieval_retry_count`. The append-only **`message_query`** trace is
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
**`strategy`** argument chooses which branches run inside `_retrieve_one`. The graph starts with
`fast_bm25_retrieval`; retry loops deterministically keep that tier until the late-interaction
threshold, then use `fast_bm25_late_interaction_retrieval`.

For multiple active retrieval queries (e.g. from `query_splitter` or `query_expansion`), the retriever runs
each query concurrently with the same per-request strategy. Each sub-query returns up to
`RETRIEVAL_TOP_K` hits from Qdrant. Results are merged in query order: deduplicate by Qdrant point id
(first occurrence wins), then return the combined list. There is no cross-query RRF and no global
`[:top_k]` cap on the merged result.

Example: `RETRIEVAL_TOP_K=8` with three active queries yields up to 24 documents per retrieval pass
(fewer if the same point id appears in more than one sub-query list). The graph's `retrieval_node`
then deduplicates by passage text and appends new rows to `retrieved_documents` (including across
retry loops in the same user turn).

## Environment Configuration

Copy `.env.example` to `.env`. Key flags:

```env
# Retrieval Strategy
USE_BM25=true
USE_LATE_INTERACTION=true
USE_MMR=true
RETRIEVAL_CANDIDATE_LIMIT=100
# Per sub-query; multi-query passes return up to RETRIEVAL_TOP_K × num(active_retrieval_queries) (before id dedup)
RETRIEVAL_TOP_K=8

# Context Management
MESSAGE_SUMMARY_TOKEN_THRESHOLD=100000
MESSAGE_SUMMARY_KEEP_RECENT=10

# Retry Loops
RETRIEVAL_LOOP_MAX_RETRIES=3

# Node Models
QUERY_NORMALISATION_MODEL=gpt-4.1-mini
QUERY_COMPLEXITY_MODEL=gpt-4.1-mini
QUERY_REWRITER_MODEL=gpt-4.1-mini
RECALL_CHECK_MODEL=gpt-4.1-mini
INTENT_CHECK_MODEL=gpt-4.1-mini
FINAL_ANSWER_MODEL=gpt-4.1-mini
QUERY_DECOMPOSITION_MODEL=gpt-4.1-mini
QUERY_EXPANSION_MODEL=gpt-4.1-mini
GAP_FILL_MODEL=gpt-4.1-mini
INTENT_CORRECTION_REWRITER_MODEL=gpt-4.1-mini

# Observability (when false: no Langfuse spans or network traffic)
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

data: {"type":"node","node":"query_complexity","status":"completed","label":"Classifying query complexity","complexity":"comparison_query","explanation":"..."}

data: {"type":"node","node":"query_splitter","status":"completed","label":"Preparing retrieval queries","active_retrieval_queries":["Enterprise refund policy","Consumer refund policy"]}

data: {"type":"node","node":"retrieval","status":"completed","label":"Retrieving documents","retrieval_strategy":"fast_bm25_retrieval","retrieval_queries":["Enterprise refund policy","Consumer refund policy"],"new_retrieved_documents":[{"score":0.42,"text":"..."}],"retrieved_doc_count":8}

data: {"type":"node","node":"recall_check","status":"completed","label":"Checking recall","recall_sufficient":true,"required_facts":{"fact1":"Enterprise refund policy terms","fact2":"Consumer refund policy terms"},"fact_verifications":{"fact1":{"fact":"Enterprise refund policy terms","evidence_available":true,"evidence_documents":["..."]},"fact2":{"fact":"Consumer refund policy terms","evidence_available":true,"evidence_documents":["..."]}},"unsupported_fact_keys":[],"retrieval_retry_count":0}

data: {"type":"final","node":"answer","status":"completed","label":"Answer ready","answer":"...","sources":[],"confidence":"high"}

data: {"type":"node","node":"clear_turn_trace","status":"completed","label":"Clearing turn trace"}

data: {"type":"done","session_id":"demo-stream","retrieved_docs":[{"score":0.42,"text":"..."}]}
```

The closing `done` frame includes `retrieved_docs` (compact `score` / `text` rows, matching `/run`)
so streaming clients can show passage counts or tooling without a second `/run` invoke.

If retry budgets are exhausted, the final event comes from `partial_answer` instead of `answer`.
Retrieval `node` frames include `new_retrieved_documents` and `retrieved_doc_count` for **this
retrieval pass only** (not the full accumulated corpus). With multiple `active_retrieval_queries`,
those counts reflect the per-query top-k merge (up to `RETRIEVAL_TOP_K` per query), not a single
global top-k. The terminal `done` frame carries all compact docs accumulated across retry loops in the turn.

## Output validation

Recall and intent rules are enforced in `output_validation/`:

- `RequiredFactsResult`: `facts` must be keyed as contiguous `fact1`, `fact2`, ...
- `RecallVerifyResult`: every fact must echo exactly; `evidence_documents` is non-empty when
  `evidence_available` is true and empty when false.
- `IntentCheckResult`: per-fact `intent_mismatch_details` is required only when
  `intent_aligned` is false.
- `GapFillResult` / `IntentCorrectionRewriteResult`: exactly three retrieval queries per fact.

### Retrieval retry tiers

```text
fast_bm25_retrieval → fast_bm25_late_interaction_retrieval
```

## Development Notes

- **Prompts**: Centralized in `prompts/`, one system prompt per graph node.
- **Output Validation**: Structured node outputs live in `output_validation/` and use Pydantic
  models with descriptive fields (recall / intent / tier rules above).
- **LLM Client**: Unified client construction in `middleware/llm_client.py` handles model
  configuration, structured output wrappers, and rate limiting.
- **Metadata Handling**: Final answer messages preserve provider metadata in `messages`. Prompt
  contexts are built as plain text or JSON so LangChain response metadata is not sent back to LLMs.
- **Observability**: When `LANGFUSE_TRACING_ENABLED=true`, each `/run` or `/run/stream` creates one
  Langfuse trace rooted at `run` or `stream_run` (`session_id` via `propagate_attributes`). Graph
  nodes emit spans with `output` only; LLM steps emit `{node}-llm` generations with **model**, **input**
  (input token count), **output** (output token count), and automatic latency. No LangChain
  `CallbackHandler`. When `false`, tracing is fully off. See `observability/langfuse_handler.py` and
  retrieval logging in `graph/graph.py` (`retrieval_node`).

  **Recall / intent spans** (minimal): `recall_check`, `intent_check`, `gap_fill`,
  `intent_correction_rewriter`, `strategy_upgrade` — flags and counts only (no full passage text).

  **Retrieval span** (`name="retrieval"`): `output` includes:

  | Field | Meaning |
  |--------|---------|
  | `strategy` | Active `retrieval_strategy` tier for this pass |
  | `queries` | `active_retrieval_queries` used (or fallback from `normalized_query`) |
  | `new_doc_count` | Count of compact docs returned this pass (before cross-turn text dedup) |
  | `rows_to_add` | Unique `{score, text}` rows appended this pass (after dedup vs prior corpus) |
  | `retrieved_documents` | Full accumulated corpus after merge (`prior + rows_to_add`) |

  Full passage text in traces can be **large** on retry loops or long chunks; Langfuse UI may be slower.
  SSE `/run/stream` events are **unchanged** (still expose `new_retrieved_documents` and counts for the
  Dash UI, not the Langfuse-only accumulated payload).
- **Code comments**: Python modules use module docstrings, `# --- section ---` headers, and inline notes
  for non-obvious routing, dedup, retries, and Langfuse branches. Prompt bodies in `prompts/` stay
  uncommented inside `SYSTEM_PROMPT` strings; each prompt file’s module docstring links prompt → graph
  node → `output_validation` schema.
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
