# Factline — Interview Prep Deep Dive

> This document is a from-scratch, top-to-bottom walkthrough of the codebase for interview prep.
> It explains **what** the system does, **why** it's built this way, **how** every module works,
> and lists **loose ends / weak spots** you should look at, fix, or at least be ready to discuss.
>
> No source code was changed while producing this document.

---

## 1. Elevator pitch (30-second answer)

> "Factline is a fact-first Retrieval-Augmented Generation (RAG) pipeline built with LangGraph,
> Qdrant, and OpenAI. Instead of doing one retrieval pass and hoping the answer is grounded, it
> **decomposes the question into atomic, checkable facts**, retrieves evidence, **verifies** that
> the retrieved passages actually support each fact (a 'recall check'), and if some facts aren't
> supported, it runs a **repair loop**: generate new targeted queries, escalate the retrieval
> strategy (BM25 → hybrid → ColBERT re-ranking), and retrieve again — excluding chunks already
> seen. Only once recall is judged sufficient (or the retry budget is exhausted) does it generate
> a final answer strictly from the retrieved documents. It's exposed via a FastAPI service with
> both a blocking endpoint and an SSE streaming endpoint, has a Dash chat UI, an isolated PMC
ingestion pipeline, and a HotpotQA + RAGAS benchmarking suite."

If asked "what's novel about it compared to vanilla RAG?" — the answer is: **verification before
generation**, not after. Most RAG systems retrieve once and generate; hallucination is caught (if
at all) after the fact via a grading/critic step. This pipeline gates the *generation* step itself
behind a recall-sufficiency check, and treats insufficient recall as a retrieval problem to repair,
not a generation problem to caveat.

---

## 2. Tech stack

| Layer | Choice | Why it matters |
|---|---|---|
| Orchestration | **LangGraph** (`StateGraph`) | Explicit state machine with conditional routing, per-node retry policies, and built-in checkpointing (multi-turn memory) |
| Vector DB | **Qdrant** (async client) | Native support for dense + sparse (BM25) + multi-vector (ColBERT) in one collection, server-side RRF fusion, MMR, and `HasId` filtering |
| LLM provider | **OpenAI** via `langchain-openai` | Structured outputs (`with_structured_output`) bound to Pydantic schemas for every node |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) | Dense vector |
| Late interaction | **Jina ColBERT v2** (`jina-colbert-v2`, 128-dim multi-vector) | Token-level re-ranking, used only as an escalation tier |
| API | **FastAPI** | `POST /run` (blocking), `POST /run/stream` (SSE), `POST /resume` (stub) |
| UI | **Dash** | Server-rendered layout + clientside JS (`ui/assets/rag_ui.js`) that consumes the SSE stream directly from the browser |
| Observability | **Langfuse** (optional) | One trace per turn, nested spans per node, nested `{node}-llm` generation spans with token counts |
| Checkpointing | `InMemorySaver` or `AsyncPostgresSaver` | Keyed by `session_id` == LangGraph `thread_id` |
| Ingestion | Unstructured.io (hi-res partition + `chunk_by_title`) | PMC PDF → structured chunks → optional LLM enrichment → Qdrant |
| Benchmarking | HotpotQA (distractor) + **RAGAS** | Context precision/recall, faithfulness, answer correctness |
| Settings | `pydantic-settings` (3 independent `.env` files) | App root, `ingestion/.env`, `benchmarking/hotpotqa/.env` — deliberately isolated |

---

## 3. Repository map

```
graph/                LangGraph state, nodes, routing (the heart of the system)
retriever/            Qdrant hybrid retriever (dense / BM25 / RRF / ColBERT)
prompts/              One system prompt per LLM node (plain triple-quoted strings)
output_validation/    Pydantic schemas — one structured-output contract per LLM node
middleware/           LLM client factory, OpenAI rate limiter, optional context truncation
observability/        Langfuse client + span helpers
tool_wrappers/        Small pure-function helpers shared by graph + benchmarking
api/                  FastAPI app: /run, /run/stream, /resume
ui/                   Dash chat client (server layout + clientside SSE JS)
config/               Pydantic Settings for the production app (.env at repo root)
ingestion/            Self-contained PMC pipeline (its own ingestion/.env, no imports from graph/)
benchmarking/hotpotqa/  Offline retrieval + graph evaluation against HotpotQA, scored with RAGAS
scripts/plot_langgraph.py  Regenerates artifacts/langgraph.{mmd,png}
artifacts/            Generated topology diagram (gitignored except README)
start.sh / stop.sh    Dev convenience scripts (uvicorn + dash)
```

**Isolation principle** (stated in the README and enforced by convention, not code): `ingestion/`
never imports from `graph/`, `retriever/`, or `benchmarking/`. Each of the three "environments"
(production app, ingestion, benchmarking) has its **own** `.env` file and its own settings class.
This means, for example, that `QUERY_DECOMPOSITION_MODEL` used for benchmark **enrichment** comes
from the *root* `.env`, while `HOTPOTQA_MAX_QUESTIONS` comes from `benchmarking/hotpotqa/.env` —
a subtlety worth remembering if asked "where does config X come from?"

---

## 4. The graph: end-to-end request walkthrough

### 4.1 Topology

```
query_normalisation
     │
     ▼
fact_decomposition
     │
     ▼
query_complexity ──(1 fact)──────────────► retrieval
     │
   (>1 fact)
     │
     ▼
query_splitter ─────────────────────────► retrieval
                                              │
                                              ▼
                                        recall_check
                                       /      |      \
                          (sufficient)  (retries left, unsupported)  (retries exhausted)
                              │                │                          │
                              ▼                ▼                          ▼
                            answer   create_queries_for_unsupported_facts  partial_answer
                              │                │                          │
                              ▼                ▼                          │
                             END        strategy_upgrade                  │
                                              │                           │
                                              ▼                           │
                                          retrieval (loop back)           │
                                                                          ▼
                                                                         END
```

Every LLM/retrieval node also has a `RetryPolicy` (transient-failure retries) with an
`error_handler=handle_node_failure` that, on final failure, redirects to `error_answer → END`
instead of crashing the whole request (**except** `answer` and `partial_answer` — see §9, Loose
Ends).

### 4.2 Node-by-node

| Node | Reads | Writes | LLM / Schema | Routes to |
|---|---|---|---|---|
| `query_normalisation` | `messages`, `message_summary` | `normalized_query` | `QUERY_NORMALISATION_MODEL` → `QueryNormalisationResult` | `fact_decomposition` (fixed) |
| `fact_decomposition` | `normalized_query` | `facts` (unified fact list) | `QUERY_DECOMPOSITION_MODEL` → `RequiredFactsResult` | `query_complexity` (fixed) |
| `query_complexity` | `facts` | `needs_split`, `retrieval_strategy="fast_bm25_retrieval"`, `active_retrieval_queries=[normalized_query]` | **none — deterministic**, `len(facts) > 1` | `query_splitter` or `retrieval` |
| `query_splitter` | `normalized_query`, `facts` | `active_retrieval_queries` (one query per fact, ideally) | `QUERY_DECOMPOSITION_MODEL` → `QuerySplitResult` | `retrieval` (fixed) |
| `retrieval` | `active_retrieval_queries`, `retrieval_strategy`, `retrieved_point_ids` (for exclusion) | `retrieved_documents` (delta, `operator.add`), `retrieved_point_ids` (delta) | none (calls `Retriever.retrieve`) | `recall_check` (fixed) |
| `recall_check` | `facts`, `retrieved_documents` | `facts` (verification fields updated), `recall_sufficient` | `RECALL_CHECK_MODEL` → `RecallVerifyResult` (**one batched call**, see §9) | `answer` / `partial_answer` / `create_queries_for_unsupported_facts` |
| `create_queries_for_unsupported_facts` | unsupported subset of `facts` | `facts` (search_queries + explanation attached), `active_retrieval_queries` | `GAP_FILL_MODEL` → `GapFillResult` | `strategy_upgrade` (fixed) |
| `strategy_upgrade` | `retrieval_retry_count` | `retrieval_retry_count += 1`, possibly escalates `retrieval_strategy` | none (deterministic) | `retrieval` (fixed — loop back) |
| `answer` | `normalized_query`, `retrieved_documents` | `messages` += `AIMessage(name="answer_node")` | `FINAL_ANSWER_MODEL` → `FinalAnswer` | `END` |
| `partial_answer` | `normalized_query`, `facts`, `retrieved_documents` | `messages` += `AIMessage(name="partial_answer_node")` | `FINAL_ANSWER_MODEL` → `FinalAnswer` | `END` |
| `error_answer` | `graph_failure` | `messages` += deterministic fallback `AIMessage` | none | `END` |

### 4.3 The "unified fact" record

Created once by `fact_decomposition`, mutated in place by `recall_check` and
`create_queries_for_unsupported_facts`. This is the backbone data structure of the whole turn:

```json
{
  "fact_id": 1,
  "fact": "Whether Enterprise tier has a published refund policy",
  "verification_status": false,
  "evidence_document_ids": [],
  "search_queries": [],
  "gap_fill_explanation": ""
}
```

`fact_id` is stable for the whole turn (assigned once, 1-based). This is what lets
`create_queries_for_unsupported_facts` and `recall_check` "talk about the same fact" across
multiple repair passes without re-matching on fuzzy text.

### 4.4 The repair loop in detail

This is the mechanism most likely to come up in an interview ("how do you avoid hallucination /
handle insufficient context?").

1. `recall_check` runs **one batched LLM call** (`verify_all_facts`) that, for every fact, asks:
   *"do the retrieved passages ALONE let a diligent reader infer this fact, without outside
   knowledge or guessing? Multi-hop chaining is only allowed if the bridge is explicit in the
   text."* Each fact gets `verification_status` (bool) and `evidence_document_ids` (catalog
   point ids — schema-enforced non-empty iff supported; membership checked against
   `document_catalog` after the LLM returns).
2. `route_after_recall_check` is a **first-match-wins** decision:
   - all facts supported → `answer`
   - `retrieval_retry_count >= RETRIEVAL_LOOP_MAX_RETRIES` (default 3) → `partial_answer`
   - otherwise → `create_queries_for_unsupported_facts` (repair)
3. `create_queries_for_unsupported_facts` asks the gap-fill LLM for **exactly 3** new search
   queries per unsupported fact (schema-enforced via `ensure_three_search_queries`, which pads
   with fallback strings if the LLM under-delivers). Query design is prescribed in the prompt:
   query 1 = entity-anchored, query 2 = BM25/keyword-friendly, query 3 = alternate phrasing/synonym.
4. `strategy_upgrade` increments `retrieval_retry_count` and **may escalate** the retrieval tier
   from `fast_bm25_retrieval` to `fast_bm25_late_interaction_retrieval` (ColBERT re-ranking) once
   `retry_count >= max(0, RETRIEVAL_LOOP_MAX_RETRIES - 2)` — i.e. escalates on the *last* couple
   of attempts, not immediately, and only if `USE_LATE_INTERACTION=true` **and** `JINA_API_KEY`
   is set.
5. Back to `retrieval`: the same Qdrant collection is queried, but with a `HasId` `must_not`
   filter built from **every point id already seen this turn** (`retrieved_point_ids`,
   accumulated via the `operator.add` reducer). This is the key repair mechanic — **it doesn't
   widen top-k; it excludes previously-seen chunks so new queries surface fresh evidence.**
6. Loop back to `recall_check`. Repeat until sufficient or budget exhausted.

**Why exclude-by-id instead of widen-top-k?** Widening top-k just re-ranks the same corpus
subset with more candidates, which doesn't fix a *phrasing* gap (bad query) — it fixes a
*recall-depth* gap. The gap-fill queries are meant to fix phrasing; the exclusion filter is what
guarantees the new queries actually surface *different* chunks instead of just re-finding the
same non-supporting passages the first pass already retrieved.

---

## 5. State management & checkpointing

`RetrievalState` (TypedDict) has two categories of keys:

- **Conversation-persistent** (`messages`, `message_summary`) — survive across `/run` calls on
  the same `thread_id` (= `session_id`). `messages` uses LangGraph's `add_messages` reducer.
- **Turn-local scratch** (`normalized_query`, `facts`, `retrieved_documents`,
  `retrieved_point_ids`, `retrieval_strategy`, `active_retrieval_queries`, `needs_split`,
  `recall_sufficient`, `retrieval_retry_count`, `graph_failure`) — wiped at the start of **every**
  `/run` / `/run/stream` call via `prepare_state_for_next_question`.

Two of the scratch keys use `operator.add` as their reducer (`retrieved_documents`,
`retrieved_point_ids`) so that multiple retrieval passes *within one turn* accumulate instead of
overwrite. But since reducers apply on every merge, including the *next turn's* reset, the reset
function uses LangGraph's `Overwrite([])` sentinel — not a bare `[]` — to force-replace the
accumulated list rather than append an empty list to it (which would be a no-op with `operator.add`
and leak the prior turn's documents into the new one). **This is a subtle but important
correctness detail** — if you removed `Overwrite` and used plain `[]`, retrieved documents would
silently accumulate forever across turns.

Only the **final** `AIMessage` per turn (answer/partial_answer/error_answer, JSON-serialized via
`build_node_ai_message`) is written to `messages`. Intermediate node outputs (facts, retrieved
docs, normalized query) are **not** persisted to the long-lived checkpoint — they're turn-scratch
only. This keeps the Postgres/InMemory checkpoint small even for long conversations.

`build_node_ai_message` preserves `additional_kwargs`, `response_metadata`, and `usage_metadata`
from the raw LangChain `AIMessage` for observability, and tags `additional_kwargs["node"]` so the
API layer can identify which terminal node produced the message.

---

## 6. Retrieval layer (`retriever/retriever.py`)

### 6.1 Strategy matrix

| Strategy | What happens | Used by |
|---|---|---|
| `fast_retrieval` | Single dense query (optionally MMR-wrapped) | Only reachable via benchmarking CLI `--strategy`; **never selected by the production graph** |
| `keyword` | BM25 sparse search only, no dense/fusion | Same — benchmarking-only in practice |
| `fast_bm25_retrieval` | Dense + BM25 **prefetch**, fused server-side with Qdrant's native RRF (`models.FusionQuery(fusion=Fusion.RRF)`) | Default first-pass and default repair-pass strategy in the graph |
| `fast_bm25_late_interaction_retrieval` | Same RRF-fused candidate pool, then re-scored with ColBERT multi-vector similarity (MaxSim) via `using=colbert` | Escalation tier during repair, gated on `USE_LATE_INTERACTION` + `JINA_API_KEY` |

### 6.2 Multi-query mechanics

One `retrieve()` call can take **multiple query strings** (e.g. one per fact from
`query_splitter`, or 3 gap-fill queries per unsupported fact). To keep this efficient:

1. Dense embeddings for **all** queries are batched into **one** OpenAI call.
2. ColBERT embeddings (if needed) are batched into **one** Jina call (max concurrency 2 via a
   semaphore, with exponential backoff on HTTP 429, respecting `Retry-After`).
3. Each query then runs its **own** Qdrant `query_points` call, but these run concurrently,
   bounded by `RETRIEVAL_SUBQUERY_MAX_CONCURRENCY` (default 8) via `asyncio.Semaphore`.
4. Results are flattened in **query order** with dedup by point id (first occurrence wins) — note
   there is **no cross-query RRF** at this stage and **no global `top_k` cap** on the merged list.
   If you split into 5 fact-level queries at `top_k=8`, you can end up with up to 40 documents fed
   into `recall_check` / `answer`. This is called out explicitly in the retriever's own docstring
   — worth knowing if asked about token-cost or latency scaling with fact count.

### 6.3 MMR and RRF

- **MMR** (`use_mmr=true`): applied only at the dense prefetch stage, using Qdrant's native
  `models.Mmr(diversity=..., candidates_limit=dense_mmr_limit * 3)` — trades relevance for
  diversity among the dense candidates *before* fusion with BM25.
- **RRF** (Reciprocal Rank Fusion): fuses the dense-candidate list and the BM25-candidate list
  into one ranked list, done entirely server-side by Qdrant (`Fusion.RRF`), not implemented in
  application code.
- **HasId exclusion**: `models.Filter(must_not=[models.HasIdCondition(has_id=clean_ids)])` is
  applied at **every prefetch and the outer query** so excluded ids can't leak back in through
  either the dense or BM25 branch.

---

## 7. Prompts & structured outputs

Every LLM node pairs one system prompt (plain string in `prompts/`) with one Pydantic schema
(`output_validation/`), invoked via `llm.with_structured_output(schema, include_raw=True)`. This
returns `{"parsed": <Model instance>, "raw": <AIMessage>}` — `raw` is kept around purely for
Langfuse token-count reporting (`update_llm_generation`).

Key validation logic worth knowing:

- `RequiredFactsResult` / `GapFillResult` / `RecallVerifyResult` all reuse
  `validate_unique_fact_texts` (a plain function, not a mixin) to reject duplicate fact strings —
  because downstream code matches facts by exact text equality in a couple of places, duplicates
  would silently corrupt merges.
- `VerifiedFact` enforces a **coupling rule** at the Pydantic level:
  `verification_status=True` ⟺ `evidence_document_ids` non-empty. This means the LLM literally
  cannot claim "supported" with no evidence ids, or cite ids while claiming unsupported —
  validation raises and the whole `ainvoke` fails (triggering the node's `RetryPolicy`).
  After merge, code also raises if any evidence id is missing from `document_catalog`.
- `GapFillFact.ensure_three_search_queries` deterministically pads to exactly 3 queries using
  fallback templates (`fact`, `"{fact} documents passages"`, `"{fact} keyword search"`, then
  numbered "alternate phrasing N") if the LLM returns fewer than 3 — so the contract "exactly 3
  queries per unsupported fact" is enforced in code, not just prompted.
- All answer-producing prompts (`final_answer`, `partial_answer`) hard-code
  **`sources` field MUST be []** — see §9, this means citation/source-attribution is a stub today.

---

## 8. Supporting infrastructure

### 8.1 `middleware/llm_client.py`
Factory for `ChatOpenAI` instances. Every call passes through a **shared, module-level**
`InMemoryRateLimiter` (`middleware/llm_rate_limit.py`) — a token-bucket limiter
(`requests_per_second`, `check_every_n_seconds`, `max_bucket_size`) instantiated once at import
time so *all* graph nodes across *all* concurrent requests share the same budget. This protects
against bursty concurrent turns (e.g. many parallel fact verifications) tripping OpenAI rate
limits.

### 8.2 `middleware/context_editing.py`
Fully implemented long-context compaction (summarize evicted turns + emit `RemoveMessage` ops to
shrink the checkpoint), but it's **wired off** in `query_normalisation_node` — the call is
commented out and replaced with `kept_messages = messages` (no truncation). Safe to enable for
long-running threads; currently no thread ever gets summarized/truncated regardless of length.

### 8.3 `observability/langfuse_handler.py`
No `CallbackHandler`, no `@observe` decorator — this codebase uses Langfuse's newer
`start_as_current_observation` context-manager API directly inside each node, gated end-to-end
by `settings.langfuse_tracing_enabled` (when `False`, `get_langfuse_client()` returns `None` and
`get_client()` from the Langfuse SDK is never even called — zero overhead when disabled). One root
span per `run`/`stream_run`/`resume`, with `propagate_attributes(session_id=...)` so all nested
spans/generations are tagged for filtering in the Langfuse UI. Token counts are pulled from
LangChain's own `usage_metadata` on the raw `AIMessage`, not from a separate accounting layer.

### 8.4 `tool_wrappers/`
Two small, pure, dependency-free modules shared between the graph and the benchmarking suite:
- `prompt_plain.py` — renders LangChain messages as clean text (no LangChain metadata noise) for
  LLM context blocks.
- `retrieval_payload.py` — `compact_documents_for_llm` always prefers
  `payload.additional_metadata.raw_text` over `payload.text`, because `payload.text` is the
  *enriched* string used only for embedding (title + summary + facts + keywords + passage), and
  feeding that enriched string to the LLM nodes would leak synthetic structure into the answer
  context. This is the single most important invariant in the whole payload contract.

---

## 9. Data contract: the Qdrant chunk payload

Both `ingestion/` (PMC) and `benchmarking/hotpotqa/` (HotpotQA) produce **the exact same payload
shape**, so the graph code doesn't need to know which corpus it's querying:

```json
{
  "text": "embedded string; equals raw_text when enrichments is {}",
  "enrichments": { "summary": "optional structured enrichment" },
  "additional_metadata": {
    "raw_text": "mandatory original passage or chunk",
    "source": "hotpotqa | pmc",
    "context_id": "source-specific keys as needed"
  }
}
```

Rule: **`text` is only ever used for embedding** (dense/BM25/ColBERT at upload time). **Every**
graph LLM node and every RAGAS metric reads `additional_metadata.raw_text` exclusively. When
enrichment is skipped (`--no-enrich`), `enrichments = {}` and `text == raw_text` — no special-casing
needed downstream.

Optional enrichment (`--enrich`) calls an LLM per chunk to produce `predicted_title`, a two-line
`summary`, deduped `keywords`, and a list of `{fact, fact_question}` pairs, which are woven into
the embedded `text` string (title → summary → facts → sample questions → keywords → passage) to
improve retrieval recall — this is a distinct, upload-time use of "facts," not to be confused with
the graph's runtime fact-decomposition.

---

## 10. API layer (`api/main.py`)

- **`POST /run`** — blocking; returns `{session_id, interrupted, question, answer, sources,
  confidence, retrieved_docs}`. `get_api_response` searches `messages` (reversed) for the first
  message named `answer_node`/`partial_answer_node`/`error_answer_node`, parses its JSON content
  into a `FinalAnswer`, and falls back to "last non-empty AI message as low confidence" if that
  fails — defense against a malformed or missing terminal message.
- **`POST /run/stream`** — SSE. `get_stream_event` maps each raw LangGraph `stream_mode="updates"`
  chunk (keyed by node name) into a stable, versioned event shape the Dash JS depends on. Only
  **counts** are streamed for retrieval (`retrieved_doc_count`, `new_doc_count`), never passage
  text — full passages are only in Langfuse (when enabled) or the final `/run` response.
- **`POST /resume`** — accepts `Command(resume=value)`; exists for a future human-in-the-loop
  `interrupt()` flow. **No node in the current graph ever calls `interrupt()`**, so this endpoint
  is unreachable in practice today (see §11).
- Lifespan (`asynccontextmanager`) builds exactly one `RetrievalGraph` per process, choosing
  `InMemorySaver` or `AsyncPostgresSaver` based on `CHECKPOINTER_USE_POSTGRES`, and cleanly closes
  the Qdrant client (and Postgres pool) on shutdown.
- CORS is scoped to loopback-style origins only (127.0.0.1/localhost/::1/0.0.0.0, any port) via a
  regex — appropriate for a local dev UI, would need tightening for a real deployment.

---

## 11. UI layer (`ui/`)

Dash app with a **server-rendered layout** and **clientside JavaScript** doing the actual chat
work: `submit_message` (in `ui/assets/rag_ui.js`, not read in depth here but referenced by the
Dash app) opens the SSE connection directly from the browser to `POST /run/stream` and updates an
inline "thinking" panel per node event, then finalizes the chat bubble on the `final`/`done`
frames. `session-id` is a client-side UUID stored in `dcc.Store`, and doubles as the LangGraph
`thread_id` — a "New chat" button rotates it and bumps `session-gen` to trigger clientside cleanup.
`RagApiClient` (`ui/api_client.py`) is a synchronous `requests`-based client used for
tests/scripts, not by the Dash UI itself (the UI talks to the API from the browser).

---

## 12. Ingestion pipeline (`ingestion/`)

Three-stage, fully isolated pipeline (own `.env`, no imports from `graph`/`retriever`/`benchmarking`):

```
download_raw_pdfs.py  →  unstructured_pipeline.py  →  chunks.json  →  upload_qdrant_embedding.py  →  Qdrant
```

1. **Download** (`download_raw_pdfs.py`): queries NCBI PMC Entrez `esearch`/`esummary` for
   open-access papers, resolves OA links (`pdf` or `tgz` package) via the PMC OA service, handles
   legacy FTP path rewriting (`oa_pdf`/`oa_package` → `deprecated`), extracts the first PDF from a
   `.tgz` when needed, and writes a `manifest.json` recording download status
   (`downloaded`/`skipped`/`failed`) per paper — a durable audit trail for step 2.
2. **Partition + chunk** (`unstructured_pipeline.py`): submits PDFs in batches to the Unstructured
   **on-demand Jobs API** (hi-res layout, optional OpenAI image description, `chunk_by_title`
   chunking with configurable overlap), polls for completion, decodes `orig_elements` (base64+zlib
   JSON), extracts embedded images separately, and strips image bytes back out of the normalized
   JSON before writing `chunks.json` (keeps the file lean while preserving image metadata
   separately).
3. **Upload** (`upload_qdrant_embedding.py`): optional LLM enrichment (`--enrich`/`--no-enrich`,
   mutually exclusive, required flag), then **deletes and recreates** the target collection
   (dense + optional ColBERT multivector + optional BM25 sparse vectors), batches OpenAI + Jina
   embedding calls, and upserts. Point ids are **deterministic** (`uuid5(NAMESPACE_URL,
   f"{filename}:{element_id}")`) so re-running upload is idempotent per element.

Notable safety details: `clear_output_dir` wipes `raw_pdfs/` before every download run (except
`.gitkeep`) — re-running `download_raw_pdfs.py` is destructive to previously downloaded PDFs by
design (a fresh query = a fresh corpus).

---

## 13. Benchmarking (`benchmarking/hotpotqa/`)

Two independent eval modes against HotpotQA **distractor** validation split:

- **`--mode retrieval`**: retrieves once per question with a given `--strategy`, scores
  **context precision** and **context recall** with RAGAS.
- **`--mode graph`**: runs the **full production graph** once per question (fresh `session_id`
  per question so no cross-question conversation bleed), scores context precision/recall,
  **faithfulness**, and **answer correctness**.

`--experiment-name` is mandatory and refuses to overwrite an existing results directory
(`assert_experiment_results_dir_available`) — an explicit anti-footgun so you can't silently
clobber a prior run's numbers. Outputs: `results.json` (raw rows), `run_metadata.json` (env
snapshot + latency percentiles), `ragas_results.json`, `benchmark_report.md` (auto-generated,
includes a type×level breakdown table).

**Actual numbers on file** (20-question stratified sample, `RETRIEVAL_TOP_K=5`):

| Metric | Retrieval-only (`fast_bm25_retrieval`) | Full graph |
|---|---|---|
| Context precision | 0.5617 | 0.4032 |
| Context recall | 0.7000 | 0.7500 |
| Faithfulness | — | 0.7993 |
| Answer correctness | — | 0.6494 |
| Partial answers | — | 4/20 (20%) |
| Mean latency | 954 ms | 31,063 ms (p95: 94,936 ms) |

Interesting discussion point if asked: **the full graph has *lower* context precision than raw
retrieval** (0.40 vs 0.56) despite *higher* context recall (0.75 vs 0.70) — consistent with the
repair loop's design: it deliberately over-retrieves across multiple passes/queries to chase
recall on hard facts, at the cost of precision (more irrelevant chunks accumulate in
`retrieved_documents` before an answer is attempted). Latency cost is very high (30s mean, up to
95s) — almost entirely repair-loop LLM+retrieval round trips on `comparison`/`hard` questions,
which score worse across the board (0.24 precision) than `bridge`/`hard` questions (0.57
precision). This is a good "what would you improve" talking point (see §14).

---

## 14. Design decisions worth being able to defend

**Q: Why decompose into facts before retrieving, instead of just retrieving on the raw question?**
Because a single retrieval pass optimized for the *whole* question often under-serves
sub-questions buried inside comparisons/multi-hop questions (e.g. "Are A and B both X?" needs
independent evidence for A and for B). Decomposing first lets the system (a) generate per-fact
queries, (b) verify per-fact recall independently, and (c) repair only the facts that actually
failed, rather than re-running the whole pipeline blindly.

**Q: Why verify recall *before* generating instead of just asking the LLM to say "I don't know" if
unsure?**
LLMs are unreliable self-graders under generation pressure — once they're mid-generation,
plausible-sounding completions are easy to produce even from thin evidence. Separating verification
(a smaller, focused judgment: "is this one fact supported, yes/no, cite evidence") from generation
(open-ended synthesis) produces a harder gate. The Pydantic-level coupling
(`verification_status=True ⟺ evidence_document_ids non-empty`, plus catalog membership checks)
removes an entire class of "claimed supported but didn't cite a real passage" failures at the
schema/application layer, before the value is treated as verified.

**Q: Why fixed top-k every pass instead of progressively widening it?**
Stated explicitly in the README: "Retrieval limits are fixed every pass. Repair loops rely on new
gap-fill queries plus Qdrant HasId exclusion, not widened top-k." The theory: a failed fact usually
means the *query* didn't hit the right subspace of the corpus, not that top-k was too small — so
widening top-k mostly just re-ranks the same near-miss neighborhood. New queries + explicit
exclusion of already-seen chunks is a more targeted fix, and keeps token/latency cost from
retrieval bounded and predictable per pass.

**Q: Why escalate to ColBERT only on the last repair attempts, not from pass 1?**
Late interaction (ColBERT) is the most expensive retrieval tier (extra embedding API call to Jina,
extra Qdrant multi-vector scoring). Reserving it for later attempts means cheap BM25+dense hybrid
handles the easy majority of facts, and the expensive tier only fires for the harder residual
cases — a cost/latency-aware design, at the expense of extra round trips for genuinely hard
questions (visible in the 31s mean / 95s p95 graph latency above).

**Q: Why does turn-scratch reset every `/run` call rather than persist and diff?**
Because facts, retrieved documents, and retry counters are all specific to *answering one user
message*; if they persisted, a follow-up question would inherit stale facts/documents from an
unrelated prior question and contaminate recall_check with irrelevant "supported" facts. Only the
message history (and, if enabled, its summary) is meant to carry cross-turn context — and that's
re-processed through `query_normalisation` each turn specifically to fold in only the *relevant*
context.

**Q: Why three separate `.env` files instead of one shared config?**
Isolation of blast radius: re-running the ingestion pipeline against the wrong collection can
**delete and recreate it** (`recreate_collection`). Keeping ingestion/benchmarking config
physically separate from the production app's `.env` makes it much harder to accidentally point a
destructive ingestion run at the production Qdrant collection.

---

## 15. Failure handling model — three distinct layers (don't conflate these)

1. **Transient node failures** (`GRAPH_NODE_RETRY_MAX_ATTEMPTS`, default 3): LangGraph's built-in
   `RetryPolicy` (exponential backoff, `initial_interval=1.0`, `backoff_factor=2.0`) retries a node
   that raised (e.g. API timeout, transient validation failure). If retries are exhausted, a
   `NodeError` is raised to the graph, caught by `error_handler=handle_node_failure`, which routes
   to `error_answer` — a deterministic, non-LLM fallback message. This is about **infrastructure
   reliability**, not answer quality.
2. **The recall repair loop** (`RETRIEVAL_LOOP_MAX_RETRIES`, default 3): not a failure path at
   all — it's the intended, designed mechanism for improving answer quality when the *first*
   retrieval pass doesn't have enough evidence. Exhausting this budget routes to `partial_answer`,
   which explicitly calls out unsupported facts in its response rather than failing.
3. **Total exhaustion** (`error_answer`): only reachable via layer 1 (node retries exhausted), and
   only for nodes that registered `error_handler=handle_node_failure`. Emits a fixed user message
   ("An error occurred while processing your request. Please try again.") with `confidence="low"`
   and no sources.

---

## 16. Anticipated interview questions (rapid-fire)

- **"Walk me through what happens when I send a message."** → normalize → decompose into facts →
  route on fact count → retrieve → verify recall per fact → repair loop (new queries, escalate
  tier, exclude seen ids, retrieve again) until sufficient or budget exhausted → answer/partial
  answer strictly from retrieved text.
- **"How do you prevent hallucination?"** → Recall verification gate before generation (schema-
  enforced evidence coupling), plus explicit prompt constraints in `final_answer`/`partial_answer`
  ("do not use outside knowledge," "if evidence is thin/contradictory, say so").
- **"How does multi-turn conversation work?"** → LangGraph checkpointing keyed by `thread_id` =
  `session_id`; only `messages` (+ optional summary) persists across turns; everything else is
  per-turn scratch wiped via `Overwrite([])` at the start of each invoke.
- **"What happens if Qdrant/OpenAI is down?"** → Node-level `RetryPolicy` retries with backoff;
  on exhaustion, most nodes degrade to `error_answer`. (Caveat: `answer`/`partial_answer` do **not**
  have an `error_handler` — see loose ends.)
- **"How would you scale this to more concurrent users?"** → `graph_max_concurrency` limits
  LangGraph's internal parallel supersteps; the shared `InMemoryRateLimiter` throttles OpenAI
  calls globally per-process; for horizontal scaling you'd need `AsyncPostgresSaver` (already
  supported) so checkpoints aren't process-local, plus a shared distributed rate limiter instead
  of the current in-process token bucket.
- **"Why LangGraph over a hand-rolled state machine or a chain of LCEL runnables?"** → Native
  support for conditional routing with cycles (the repair loop is a cycle back to `retrieval`),
  per-node retry policies, and checkpointing/threading for multi-turn state — all of which you'd
  otherwise hand-roll.
- **"What's the difference between the node-retry policy and the repair loop?"** → See §15 above;
  one is infrastructure resilience, the other is answer-quality iteration. Easy to conflate; be
  precise about it.
- **"How do RRF, MMR, and ColBERT differ here?"** → RRF fuses two *independent* ranked lists
  (dense + BM25) by rank position, done server-side by Qdrant. MMR re-ranks *within* the dense
  candidate pool for diversity before fusion. ColBERT is a late-interaction re-ranker applied
  *after* RRF fusion, using token-level MaxSim similarity on a small candidate pool — it's the
  most accurate and most expensive, so it's reserved for late repair attempts only.
- **"Why is context precision worse in the full graph than in raw retrieval, if recall is
  better?"** → Direct trade-off of the repair loop: chasing higher recall on hard facts means more
  (sometimes irrelevant) passages accumulate across multiple retrieval passes before an answer is
  attempted. Good springboard into "what would you do about it" (e.g., let `answer`/`recall_check`
  operate on a filtered subset of `retrieved_documents` per fact rather than the full accumulated
  corpus, or de-prioritize low-score docs before the final answer prompt).
- **"What's the `retrieved_point_ids` for, exactly?"** → Cross-pass memory of every Qdrant point id
  already surfaced this turn, so repair-loop retrieval calls can `HasId must_not` exclude them and
  guarantee forward progress (new evidence) instead of re-finding the same non-supporting chunks.

---

## 17. Loose ends / things worth reviewing or fixing

These are candid observations from reading the code closely — some are genuine bugs, some are
just inconsistencies or dead code worth cleaning up, all are good to have opinions on if asked.
No changes were made to any of these; this is a punch list for you to pick from.

1. **`answer_node` / `partial_answer_node` have no `error_handler`.** In
   `RetrievalGraph.build_graph`, every LLM/retrieval node registers
   `error_handler=handle_node_failure` **except** `answer` and `partial_answer` (they only get a
   `retry_policy`). If the final-answer LLM call fails repeatedly (retries exhausted), the
   exception propagates uncaught out of `graph.ainvoke` — the FastAPI route has no try/except
   around it either, so the client gets an unhandled 500 instead of the friendly
   `ERROR_ANSWER_USER_MESSAGE` that every *other* failure path produces. This is probably an
   oversight rather than intentional, since `error_answer` exists precisely to avoid this.

4. **Dead helper functions in `output_validation/retrieval_strategy.py`.**
   `retrieval_strategy_rank`, `is_strictly_heavier`, and `minimum_heavier_tier` are fully
   implemented and exported, but grep shows zero call sites anywhere in the codebase.
   `strategy_upgrade_node` reimplements its own ad-hoc escalation logic (a `late_threshold`
   comparison) instead of using these. Either wire them in (they'd make the escalation logic in
   `strategy_upgrade_node` more declarative/testable) or remove them.

5. **`QueryComplexityResult` schema is unused.** `output_validation/query_complexity.py` defines a
   full LLM-classification schema (`complexity: "simple_query" | "needs_split"` +
   `explanation`), but `query_complexity_node` is **fully deterministic** (`len(facts) > 1`) and
   never calls an LLM or uses this schema. Looks like an earlier version used an LLM classifier
   here and it was later replaced with a cheaper deterministic rule, leaving the schema orphaned.

6. **Stale generated diagram.** `artifacts/langgraph.mmd` (and presumably the matching `.png`,
   gitignored) is missing the `error_answer --> __end__` edge and doesn't show `error_answer` or
   the error-handler nodes wired to anything — it was almost certainly generated *before* the
   retry-policy/`error_answer` work was added to `graph/graph.py`. Per
   `artifacts/README.md`, regenerate it with `python scripts/plot_langgraph.py` (needs the
   `rag_env_1` venv active). Cheap fix, but a good "did you actually run this recently" tell if an
   interviewer asks you to show the diagram.

7. **Unused env var `RETRIEVAL_SUBQUERY_PARALLEL`.** Present in `.env.example` (line 45,
   commented as controlling "multi-query retrieval... parallel Qdrant"), but `config/settings.py`
   has no such field — only `retrieval_subquery_max_concurrency` exists and is always used (the
   code is unconditionally concurrent, bounded by a semaphore; there's no serial fallback path).
   Because `Settings.model_config` sets `extra="ignore"`, setting this in a real `.env` silently
   does nothing. Either remove the stale line from `.env.example` or wire up an actual toggle.

8. **Default model names differ between `config/settings.py` and `.env.example`.**
   `Settings` defaults to `gpt-5-mini` / `gpt-5.1` for various nodes, while `.env.example` suggests
   `gpt-5.4` / `gpt-5.4-mini` / `gpt-5.5`. Not a bug (env values always win when `.env` is copied),
   but worth reconciling so the "defaults if you don't set anything" story and the "recommended
   `.env.example` values" story agree — right now a fresh clone that forgets to copy `.env.example`
   silently runs on a different model tier than the one the example file recommends.

9. **No automated tests.** There is no `tests/` directory and no test runner (`pytest`, etc.) in
   `requirements.txt`. All the Pydantic validators (unique fact texts, evidence/verification
   coupling, `ensure_three_search_queries` padding logic, `add_queries_for_unsupported_facts`
   merge-by-fact-id logic) are exactly the kind of pure, deterministic logic that's cheap to unit
   test and currently has zero coverage. Good, low-risk place to add value before an interview
   ("I noticed there were no tests, so I added coverage for X").

10. **`/resume` and interrupt handling are unreachable.** `api/main.py`'s `/resume` route and the
    `RetrievalGraph.resume` method exist and are wired correctly, but **no node in the current
    graph ever calls `interrupt()`**, so `result.get("__interrupt__")` in `get_api_response` will
    always be empty and `/resume` can never be meaningfully triggered end-to-end today. Confirmed
    intentional ("Reserved for future clarification interrupts" in the docstring) — just be aware
    it's scaffolding, not a working feature, if asked to demo it.

11. **`fast_retrieval` and `keyword` strategies are effectively benchmark-only.** They're valid
    members of the `RetrievalStrategy` literal and fully implemented in `Retriever.retrieve_one`,
    but no graph node (`query_complexity_node` or `strategy_upgrade_node`) ever selects them — the
    production graph only ever uses `fast_bm25_retrieval` and
    `fast_bm25_late_interaction_retrieval`. They're reachable only via
    `benchmarking/hotpotqa/run_evaluation.py --mode retrieval --strategy ...`. Not wrong, just
    worth knowing so you don't assume the graph is exercising dense-only or BM25-only search in
    production.

12. **Context-editing/summarization is fully built but disabled.** `middleware/context_editing.py`
    (token-threshold-based truncation + LLM summarization of evicted turns + `RemoveMessage`
    cleanup) is complete and imported (commented) in `query_normalisation_node`, but the actual
    call is commented out — `kept_messages = messages` unconditionally. For very long-running
    threads, `messages` will grow unbounded in the checkpoint (Postgres or otherwise) since nothing
    ever evicts old turns today. Flip the two commented lines back on to activate it — the code
    path looks otherwise ready, though it's untested (see point 9) and hardcodes its own model
    (`gpt-4.1-mini`) rather than reading from `Settings`.

13. **Retrieval corpus can grow much larger than `RETRIEVAL_TOP_K` implies.** As noted in §6.2,
    multi-query retrieval has no cross-query cap — a 5-fact question with `query_splitter` and a
    full repair loop could plausibly push 30–40+ documents into a single `recall_check`/`answer`
    prompt. Combined with the accumulation across repair passes (`operator.add`, never trimmed
    within a turn), token cost and latency for `recall_check`/`answer` scale with fact count and
    retry count in a way that isn't obviously bounded by any single setting — worth a "how would
    you cap this" conversation (e.g. per-fact document caps, or re-ranking/truncating
    `retrieved_documents` before the final answer prompt).

---

## 18. Glossary (for quick recall under interview pressure)

| Term | Meaning here |
|---|---|
| **Fact** | An atomic, checkable information need (not an answer value), produced once per turn by `fact_decomposition` |
| **Recall (in this codebase)** | Whether retrieved passages *sufficiently support* a fact — not the IR metric "recall" directly, though related |
| **Recall-sufficient** | `True` iff **every** fact's `verification_status` is `True` |
| **Repair loop** | The cycle `recall_check → create_queries_for_unsupported_facts → strategy_upgrade → retrieval → recall_check`, bounded by `RETRIEVAL_LOOP_MAX_RETRIES` |
| **Gap-fill** | The LLM step that proposes new search queries for facts that failed recall verification |
| **Strategy escalation** | Switching `retrieval_strategy` from BM25-hybrid to ColBERT late-interaction on later repair attempts |
| **HasId exclusion** | Qdrant `must_not` filter over previously-seen point ids, used to force new evidence on repair passes |
| **Turn scratch** | State keys wiped at the start of every `/run` invoke (facts, retrieved docs, retry counters, etc.) |
| **`thread_id`** | LangGraph's checkpoint key; equals the API's `session_id` |
| **Raw text vs. enriched text** | `additional_metadata.raw_text` (original passage, used by all LLM nodes and RAGAS) vs. `payload.text` (enriched string used only for embedding) |
| **RRF** | Reciprocal Rank Fusion — Qdrant-native fusion of dense + BM25 ranked lists |
| **MMR** | Maximal Marginal Relevance — diversity-aware re-ranking within the dense candidate pool |
| **Late interaction / ColBERT** | Token-level multi-vector similarity re-ranking (MaxSim) via Jina, applied after RRF fusion on a small candidate pool |
