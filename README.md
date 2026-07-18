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
5. **Verify recall** with one parallel LLM call per fact against retrieved passages; update verification in place on the same `facts` list.
6. **Repair** unsupported facts via `create_queries_for_unsupported_facts` → `strategy_upgrade` → retrieval (until retries exhausted).
7. **Answer** (or partial answer) with `cited_document_ids` from `document_catalog`.
8. **Validate cited ids** (retry answer up to `CITED_ID_RETRY_MAX`), then **faithfulness** (retry up to `ANSWER_RETRY_MAX`); code fills `sources` from catalog.

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
| Node retries | `GRAPH_NODE_RETRY_MAX_ATTEMPTS` (LangGraph `RetryPolicy` on LLM/retrieval nodes for transient failures; separate from recall repair loop and routes to `error_answer` when exhausted) |
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

### Production ingestion

PMC PDF download, Unstructured partition/chunk, and Qdrant upload live under **`ingestion/`** with its own `.env`. See **[ingestion/README.md](ingestion/README.md)** for the full flow, commands, and env reference.

```text
download_data  →  unstructured_pipeline  →  chunks.json  →  qdrant_upload  →  Qdrant
```

**Isolation:** `ingestion/` does not import graph, retriever, or benchmark code. HotpotQA benchmarking is separate under `benchmarking/hotpotqa/`. Graph reads raw passage text via `tool_wrappers/retrieval_payload.py`.

### At retrieval time

| Layer | Shape | `text` meaning |
|-------|--------|----------------|
| Retriever hit | `{id, score, rank, payload}` | Full Qdrant payload |
| Graph `document_catalog` (during turn) | `{id: {text, source, score}}` | **`raw_text`** + `additional_metadata.source` |
| API `/run` `document_catalog` | same map | Corpus from the completed turn (reset on the next `/run`) |

Catalog build: `tool_wrappers/retrieval_payload.py` (`catalog_entries_from_retriever_hits`). After answer: `validate_cited_ids` → `faithfulness`; `sources` filled from catalog by code.

After changing the contract, **re-upload** your Qdrant collection (e.g. `python -m benchmarking.hotpotqa.qdrant_upload.upload` for benchmarks).

## API

| Endpoint | Description |
|----------|-------------|
| `POST /run` | Run one turn; returns `{ answer, sources, confidence, cited_document_ids, document_catalog }` |
| `POST /run/stream` | Same turn with SSE node progress (counts only on the wire) |
| `POST /resume` | Reserved for future clarification interrupts |

**`/run` response:** Turn-local scratch (including `document_catalog`) is reset at the start of the next `/run`. `sources` are code-filled from catalog after faithfulness (non-empty `source` labels only).

**`/run/stream`:** Node events expose retrieval **counts** (not passage text). Final frame is emitted after faithfulness (or cited-id retry exhaustion). `done` includes `retrieved_doc_count`.

Example `/run` JSON shape:

```json
{
  "session_id": "demo",
  "interrupted": false,
  "question": null,
  "answer": "...",
  "sources": ["hotpotqa"],
  "confidence": "high",
  "cited_document_ids": ["uuid..."],
  "document_catalog": {}
}
```

Example SSE frames:

```text
data: {"type":"node","node":"fact_decomposition","status":"completed","label":"Decomposing facts","fact_count":2}
data: {"type":"node","node":"recall_check","status":"completed","label":"Checking recall","recall_sufficient":true,"unsupported_fact_count":0}
data: {"type":"final","node":"faithfulness","status":"completed","label":"Answer ready","answer":"...","sources":["hotpotqa"]}
data: {"type":"done","session_id":"demo","retrieved_doc_count":12}
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
| `ingestion/` | Self-contained PMC pipeline: download, chunk stub, `.env`, embed, Qdrant upload |
| `benchmarking/` | Optional offline retrieval / eval suite |
| `artifacts/` | Graph topology (`langgraph.png`, `langgraph.mmd`); regenerate with `python scripts/plot_langgraph.py` |

## Observability

Set `LANGFUSE_TRACING_ENABLED=true` for one trace per `/run` or `/run/stream` with nested node spans and LLM generations (token counts). SSE and the UI expose counts only; full passage text stays in Langfuse when enabled.

## Stack

LangGraph · Qdrant · OpenAI · optional Jina ColBERT · optional Langfuse
