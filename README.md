# Factline

> Grounded answers from an explicit retrieval graph — decompose, verify, repair, answer.

Factline is a fact-first RAG pipeline built on **LangGraph**, **Qdrant**, and **OpenAI**. Each turn decomposes the question into checkable facts, retrieves evidence, verifies recall per fact, repairs gaps when needed, and only then answers from retrieved documents.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
uvicorn api.main:app --reload
```

Send a query:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Compare refund policies for Enterprise and Consumer tiers."}'
```

Stream node progress (SSE):

```bash
curl -N -X POST http://127.0.0.1:8000/run/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Compare refund policies for Enterprise and Consumer tiers."}'
```

Optional Dash UI (port 8050): `python ui/dash_app.py`

## How it works

![Graph topology](artifacts/langgraph.png)

1. **Normalize** the latest user message into a standalone query.
2. **Decompose** into an ordered `facts` list (`fact_id`, verification shell).
3. **Route** on fact count: one fact → retrieve with the normalized query; multiple facts → split into focused queries.
4. **Retrieve** with hybrid Qdrant search; repair passes exclude already-seen point ids (HasId) and dedupe by id.
5. **Verify recall** per fact against retrieved passages; update verification in place on the same `facts` list.
6. **Repair** unsupported facts via `gap_fill` → `strategy_upgrade` → retrieval (until retries exhausted).
7. **Answer** (or partial answer) strictly from retrieved documents, then clear turn-local state.

Unified fact record:

```json
{
  "fact_id": 1,
  "fact": "Whether Enterprise tier has a published refund policy",
  "verification_status": false,
  "verification_report": "",
  "evidence_documents": [],
  "search_queries": [],
  "gap_fill_explanation": ""
}
```

## Configuration

Copy `.env.example` to `.env`. Essential groups:

| Group | Variables |
|-------|-----------|
| Qdrant | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME` |
| Retrieval | `RETRIEVAL_TOP_K`, `RETRIEVAL_CANDIDATE_DENSE_MMR`, `RETRIEVAL_CANDIDATE_BM25`, `RETRIEVAL_CANDIDATE_FOR_LATE_INTERACTION`, `RETRIEVAL_MMR_DIVERSITY` |
| Repair | `RETRIEVAL_LOOP_MAX_RETRIES` (repair loops escalate to ColBERT late interaction when enabled; default starts at `fast_bm25_retrieval`) |
| Models | `QUERY_NORMALISATION_MODEL`, `QUERY_DECOMPOSITION_MODEL`, `RECALL_CHECK_MODEL`, `GAP_FILL_MODEL`, `FINAL_ANSWER_MODEL` |
| Observability | `LANGFUSE_TRACING_ENABLED` (+ Langfuse keys when true) |

Retrieval limits are **fixed every pass**. Repair loops rely on new gap-fill queries plus Qdrant HasId exclusion, not widened top-k.

## Chunk payload contract

Every Qdrant point (HotpotQA, PMC, or future corpora) uses the same payload shape. **Embedding** uses enriched `text`; **graph LLM nodes and RAGAS** use raw passage text only (`additional_metadata.raw_text`).

### Qdrant point `payload`

```json
{
  "text": "embedded string; equals raw_text when enrichments is {}",
  "enrichments": {
    "summary": "optional structured enrichment"
  },
  "additional_metadata": {
    "raw_text": "mandatory original passage or chunk",
    "source": "hotpotqa",
    "context_id": "source-specific keys as needed"
  }
}
```

| Field | Rule |
|-------|------|
| `text` | Only string used for dense, BM25, and ColBERT at upload |
| `enrichments` | Structured LLM enrichment; `{}` when none |
| `additional_metadata.raw_text` | Always required; used by graph LLM nodes and RAGAS |
| No enrichment | `enrichments = {}` and `text == raw_text` |

### At retrieval time

| Layer | Shape | `text` meaning |
|-------|--------|----------------|
| Retriever hit | `{id, score, rank, payload}` | Full Qdrant payload |
| Graph `retrieved_documents` (during turn) | `{id, score, text}` | **`raw_text`** via `get_raw_text()` — not enriched embed string |
| API `/run` `retrieved_docs` | `[]` | **Intentionally empty** after `clear_turn_trace`; answer is the user-facing output |

Compaction happens in `tool_wrappers/retrieval_payload.py` (`compact_documents_for_llm`). Adapters live under `ingestion/adapters/` (HotpotQA today; PMC TBD). Shared upload: `ingestion/qdrant_upload.py`. Schema helpers: `ingestion/schema.py`.

After changing the contract, **re-upload** your Qdrant collection (e.g. `python -m benchmarking.hotpotqa.qdrant_upload.upload` for benchmarks).

## API

| Endpoint | Description |
|----------|-------------|
| `POST /run` | Run one turn; returns `{ answer, sources, confidence, retrieved_docs }` |
| `POST /run/stream` | Same turn with SSE node progress (counts only on the wire) |
| `POST /resume` | Reserved for future clarification interrupts |

**`/run` response:** The graph clears turn-local scratch (including `retrieved_documents`) in `clear_turn_trace` after `answer` or `partial_answer`. The API therefore returns **`retrieved_docs: []`** by design — clients should use `answer`, `sources`, and `confidence`. Passage text during the turn lives in graph state for recall/answer nodes only; it is not persisted in the checkpoint or echoed on `/run`.

**`/run/stream`:** Node events expose retrieval **counts** (not passage text). The final `done` frame includes `retrieved_doc_count`. Full passages are available in Langfuse when tracing is enabled.

Example `/run` JSON shape:

```json
{
  "session_id": "demo",
  "interrupted": false,
  "question": null,
  "answer": "...",
  "sources": [],
  "confidence": "high",
  "retrieved_docs": []
}
```

Example SSE frames:

```text
data: {"type":"node","node":"fact_decomposition","status":"completed","label":"Decomposing facts","fact_count":2}
data: {"type":"node","node":"recall_check","status":"completed","label":"Checking recall","recall_sufficient":true,"unsupported_fact_count":0}
data: {"type":"final","node":"answer","status":"completed","label":"Answer ready","answer":"..."}
data: {"type":"done","session_id":"demo","retrieved_doc_count":12}
```

## Scripts

```bash
python check_settings.py          # list unused Settings fields (dev utility)
```

## Project layout

| Path | Role |
|------|------|
| `graph/` | LangGraph state, nodes, routing |
| `retriever/` | Qdrant hybrid retrieval (dense, BM25, ColBERT) |
| `prompts/` | One system prompt per LLM node |
| `output_validation/` | Pydantic schemas for structured outputs |
| `api/` | FastAPI `/run`, `/run/stream`, `/resume` |
| `ui/` | Dash chat client |
| `ingestion/` | Download, chunk schema, adapters, shared Qdrant upload |
| `benchmarking/` | Optional offline retrieval / eval suite |
| `artifacts/` | Graph topology diagram |

## Observability

Set `LANGFUSE_TRACING_ENABLED=true` for one trace per `/run` or `/run/stream` with nested node spans and LLM generations (token counts). SSE and the UI expose counts only; full passage text stays in Langfuse when enabled.

## Stack

LangGraph · Qdrant · OpenAI · optional Jina ColBERT · optional Langfuse
