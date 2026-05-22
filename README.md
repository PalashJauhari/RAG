# Advanced RAG Orchestration Pipeline

This project implements a production-grade Retrieval-Augmented Generation (RAG) pipeline built
on **LangGraph**, **Qdrant**, and **OpenAI**.

The system uses an explicit LangGraph flow that normalizes the user's query, decomposes required
facts, routes retrieval by fact count, retrieves evidence, runs fact-based recall check and intent
alignment, repairs gaps when needed, and only then produces a grounded answer.

## Key Features

- **Query Normalisation First**: The latest user message is rewritten into a standalone query using
  conversation context while preserving ambiguity instead of guessing.
- **Deterministic fact-count routing**: After fact decomposition, `query_complexity` routes by fact
  count only — `len(required_facts) > 1` → `needs_split` (query_splitter); otherwise `simple_query`
  (direct retrieval with the normalized query). No LLM call for this step.
- **Specialized Query Preparation**:
  - Simple queries go directly to retrieval.
  - Queries requiring multiple fact targets route to `query_splitter`, which turns stable
    `required_facts` into focused retrieval queries.
- **Fact-first recall and repair loop**: `fact_decomposition` decomposes the normalized query once
  into an ordered `[{fact: "..."}]` list before retrieval. `recall_check` only verifies those fixed
  facts with `verification_status`, `verification_report`, and verbatim evidence excerpts.
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
  On each repair pass, all four limits scale by multiplier `m = 1 + retrieval_retry_count`, each
  capped by its `_MAX`: `RETRIEVAL_TOP_K`, `RETRIEVAL_CANDIDATE_DENSE_MMR`,
  `RETRIEVAL_CANDIDATE_BM25`, and `RETRIEVAL_CANDIDATE_FOR_LATE_INTERACTION`.
- **Structured turn trace**: Append-only **`message_query`** audit rows across nodes; a terminal
  **`clear_turn_trace`** node resets the trace after each answer using LangGraph **`Overwrite([])`**
  so multi-turn threads do not leak prior-turn diagnostics into the next query.
- **Lean Message State**: `messages` stores user turns and final answer node outputs only. Node
  scratch data lives in explicit keys such as `normalized_query`, `active_retrieval_queries`,
  `retrieval_strategy`, `message_query`, `retrieved_documents`, `required_facts`,
  `verified_facts`, and `fact_intents`.
- **Strict Grounding**: The final answer node answers only from retrieved documents. Source citation
  wiring is intentionally deferred, so `sources` is currently returned as an empty array. The final
  answer prompt receives the normalized query, retrieval strategy, active retrieval queries, and
  retrieved documents. Partial answer also receives `verified_facts` and retry context.
- **Slim audit trace**: `message_query` rows avoid duplicating live state; retrieval rows carry
  `queries`, `strategy`, `rows_to_add_count`, and accumulated corpus size.

## Architecture

```text
FastAPI /run
  -> query_normalisation_node
  -> fact_decomposition_node
  -> query_complexity_node
      simple_query
        -> retrieval_node
      needs_split
        -> query_splitter_node
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
`retrieved_documents`, `required_facts`, `verified_facts`, `fact_intents`, and
`retrieval_retry_count`. The append-only **`message_query`** trace is
cleared **after** `answer` / `partial_answer` along with `retrieved_documents` by
**`clear_turn_trace_node`** using LangGraph
**`Overwrite([])`** (turn-local invokes still pass `message_query: []` with other scratch, but the
authoritative reset for reducer-backed history is the terminal clear node).

`/resume` is still exposed by the API so clarification can be reintroduced later without changing
the client contract. The current graph does not interrupt for ambiguous queries.

## Advanced Retrieval Flow

The `Retriever.retrieve(queries, strategy)` branch selects behavior **per request**:

| Strategy | Dense prefetch | BM25 prefetch | RRF output | ColBERT output |
|----------|----------------|---------------|------------|----------------|
| `keyword` | — | — | — | `RETRIEVAL_TOP_K` |
| `fast_retrieval` | MMR pool `DENSE_MMR×3` if MMR | — | — | `RETRIEVAL_TOP_K` |
| `fast_bm25_retrieval` | `RETRIEVAL_CANDIDATE_DENSE_MMR` | `RETRIEVAL_CANDIDATE_BM25` | `RETRIEVAL_TOP_K` | — |
| `fast_bm25_late_interaction_retrieval` | same prefetches | same | `RETRIEVAL_CANDIDATE_FOR_LATE_INTERACTION` | `RETRIEVAL_TOP_K` |

When `USE_MMR=true`, dense retrieval uses an internal MMR candidate pool of
`RETRIEVAL_CANDIDATE_DENSE_MMR × 3` and returns up to `RETRIEVAL_CANDIDATE_DENSE_MMR` dense hits
(hybrid path) or `RETRIEVAL_TOP_K` (`fast_retrieval` / `keyword`).

Environment flags **`USE_BM25`**, **`USE_LATE_INTERACTION`**, and **`USE_MMR`** still configure the
Qdrant client (e.g. cloud inference), embedding/MMR parameters, and Jina availability; the
**`strategy`** argument chooses which branches run inside `_retrieve_one`. The graph starts with
`fast_bm25_retrieval`; retry loops keep that tier until the late-interaction threshold, then upgrade
to `fast_bm25_late_interaction_retrieval` only when **`USE_LATE_INTERACTION=true`** and **`JINA_API_KEY`**
is set. Otherwise repair loops stay on hybrid retrieval (limits still scale; ColBERT never runs).

On repair loops, effective limits use multiplier `m = 1 + retrieval_retry_count` on each base limit,
capped by the matching `_MAX` env var.

For multiple active retrieval queries from `query_splitter`, the retriever runs
each sub-query sequentially with the same per-request strategy. Each sub-query returns up to
`effective_top_k` hits from Qdrant (base `RETRIEVAL_TOP_K` on the first pass; widens on repair
loops). Results are merged in query order: deduplicate by Qdrant point id
(first occurrence wins), then return the combined list. There is no cross-query RRF and no global
`[:top_k]` cap on the merged result.

Example: `RETRIEVAL_TOP_K=8` with three active queries yields up to 24 documents per retrieval pass
on the first try (fewer if the same point id appears in more than one sub-query list). After one
repair loop (`retrieval_retry_count=1`), each sub-query returns up to 16 hits. The graph's
`retrieval_node` then deduplicates by passage text and appends new rows to `retrieved_documents`
(including across retry loops in the same user turn).

## Environment Configuration

Copy `.env.example` to `.env`. Key flags:

```env
# Retrieval Strategy
USE_BM25=true
USE_LATE_INTERACTION=true
USE_MMR=true
# Final docs per sub-query; repair loops scale all limits × (1 + retry_count)
RETRIEVAL_TOP_K=8
RETRIEVAL_TOP_K_MAX=64
RETRIEVAL_CANDIDATE_DENSE_MMR=100
RETRIEVAL_CANDIDATE_DENSE_MMR_MAX=500
RETRIEVAL_CANDIDATE_BM25=100
RETRIEVAL_CANDIDATE_BM25_MAX=500
RETRIEVAL_CANDIDATE_FOR_LATE_INTERACTION=100
RETRIEVAL_CANDIDATE_FOR_LATE_INTERACTION_MAX=500

# Context Management
MESSAGE_SUMMARY_TOKEN_THRESHOLD=100000
MESSAGE_SUMMARY_KEEP_RECENT=10

# Retry Loops
RETRIEVAL_LOOP_MAX_RETRIES=3

# Node Models (one env var per LLM node; query_complexity is deterministic — no model)
QUERY_NORMALISATION_MODEL=gpt-5-mini          # query_normalisation
QUERY_DECOMPOSITION_MODEL=gpt-5-mini          # fact_decomposition + query_splitter
RECALL_CHECK_MODEL=gpt-5.1                    # recall_check
INTENT_CHECK_MODEL=gpt-5-mini                 # intent_check
GAP_FILL_MODEL=gpt-5-mini                     # gap_fill
INTENT_CORRECTION_REWRITER_MODEL=gpt-5-mini   # intent_correction_rewriter
FINAL_ANSWER_MODEL=gpt-5.1                    # answer + partial_answer

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
- `retrieved_docs`: Empty after terminal cleanup; use Langfuse for full decision-context documents.

### Stream Graph Progress

The existing `/run` endpoint remains unchanged. For node-level progress streaming, use:

```bash
curl -N -X POST http://127.0.0.1:8000/run/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-stream","message":"Compare the refund policies for Enterprise and Consumer tiers."}'
```

`/run/stream` returns Server-Sent Events (`text/event-stream`). Each event is a JSON object in a
`data:` frame. The stream reports graph progress after each node completes; it does not stream final
answer tokens word by word. **Langfuse traces keep full facts and passage text; SSE/UI expose counts
only** (no `required_facts`, `verified_facts`, or document bodies on the wire).

Example events:

```text
data: {"type":"node","node":"query_normalisation","status":"completed","label":"Normalizing query","normalized_query":"Compare the Enterprise refund policy with the Consumer refund policy."}

data: {"type":"node","node":"fact_decomposition","status":"completed","label":"Decomposing required facts","required_fact_count":2}

data: {"type":"node","node":"query_complexity","status":"completed","label":"Routing by fact count","complexity":"needs_split","explanation":"3 required facts → split retrieval per fact."}

data: {"type":"node","node":"query_splitter","status":"completed","label":"Preparing retrieval queries","active_retrieval_queries":["Enterprise refund policy","Consumer refund policy"]}

data: {"type":"node","node":"retrieval","status":"completed","label":"Retrieving documents","retrieval_strategy":"fast_bm25_retrieval","retrieval_queries":["Enterprise refund policy","Consumer refund policy"],"new_doc_count":8,"retrieved_doc_count":8,"retrieval_loop_count":0}

data: {"type":"node","node":"recall_check","status":"completed","label":"Checking recall","recall_sufficient":true,"unsupported_fact_count":0,"retrieved_doc_count":8,"retrieval_loop_count":0}

data: {"type":"node","node":"retrieval","status":"completed","label":"Retrieving documents","retrieval_strategy":"fast_bm25_retrieval","retrieval_queries":["Enterprise refund policy terms"],"new_doc_count":30,"retrieved_doc_count":41,"retrieval_loop_count":1}

data: {"type":"node","node":"recall_check","status":"completed","label":"Checking recall","recall_sufficient":false,"unsupported_fact_count":3,"retrieved_doc_count":41,"retrieval_loop_count":1}

data: {"type":"node","node":"strategy_upgrade","status":"completed","label":"Evaluating retrieval tier","retrieval_strategy":"fast_bm25_retrieval","retrieval_loop_count":1}

data: {"type":"final","node":"answer","status":"completed","label":"Answer ready","answer":"...","sources":[],"confidence":"high","retrieved_doc_count":41}

data: {"type":"node","node":"clear_turn_trace","status":"completed","label":"Clearing turn trace"}

data: {"type":"done","session_id":"demo-stream","retrieved_doc_count":41}
```

The closing `done` frame includes `retrieved_doc_count` only; document bodies are not sent over SSE.
`retrieval_loop_count` tracks repair passes (0 on first retrieval/recall; increments after each
`strategy_upgrade`).

If retry budgets are exhausted, the final event comes from `partial_answer` instead of `answer`.
Retrieval `node` frames include `new_doc_count` for rows added this pass and
`retrieved_doc_count` for the accumulated per-turn corpus. With multiple
`active_retrieval_queries`, retrieval still fetches up to `RETRIEVAL_TOP_K` per query before
deduplication.

## Output validation

Fact decomposition, recall, and intent rules are enforced in `output_validation/`:

- `RequiredFactsResult` (`fact_decomposition`): `facts` is an ordered list of unique
  `{"fact": "..."}` objects.
- `RecallVerifyResult`: every fact must echo exactly; `verification_report` is non-empty;
  `evidence_documents` is non-empty when `verification_status` is true and empty when false.
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

  **Fact / recall / intent spans**: `fact_decomposition` logs the normalized query and stable
  `required_facts`. `recall_check` logs `required_facts`, `verified_facts`,
  derived unsupported facts, `recall_sufficient`, and the full accumulated `retrieved_documents`
  used for the decision. Repair nodes log their full parsed per-fact outputs.

  **Retrieval span** (`name="retrieval"`): `output` includes:

  | Field | Meaning |
  |--------|---------|
  | `strategy` | Active `retrieval_strategy` tier for this pass |
  | `queries` | `active_retrieval_queries` used (or fallback from `normalized_query`) |
  | `retrieval_retry_count` | Repair-loop counter for this pass |
  | `effective_top_k` | Scaled final per-sub-query limit |
  | `effective_dense_mmr` | Scaled dense prefetch / MMR target |
  | `effective_bm25` | Scaled BM25 prefetch limit |
  | `effective_late_interaction` | Scaled RRF pool before ColBERT |
  | `candidate_doc_count` | Count of compact docs returned this pass before text dedup |
  | `rows_to_add` | Unique `{score, text}` rows appended this pass (after dedup vs prior corpus) |
  | `rows_to_add_count` | Count of rows appended this pass |
  | `corpus_size_before` | Accumulated corpus size before this pass |
  | `corpus_size_after` | Accumulated corpus size after this pass |

  Full accumulated passage text is logged on `recall_check`, `answer`, and `partial_answer` spans
  because those nodes make decisions from that corpus. SSE `/run/stream` events expose counts only.
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
