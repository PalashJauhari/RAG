# Ingestion (arXiv → Qdrant)

Standalone arXiv pipeline: download PDFs, partition/chunk with Unstructured, embed and upsert to Qdrant.

Configure via **`ingestion/.env` only** — not repo root or HotpotQA benchmark env.

Target collection name: **`arxiv_cs_ds`**.

## Prerequisites

- Python env with repo dependencies (e.g. `rag_env_1`)
- `cp ingestion/.env.example ingestion/.env` and fill keys
- Unstructured image LLM: configure OpenAI in [platform.unstructured.io](https://platform.unstructured.io) (not in `.env`)

Run all commands from the **repo root**.

## Steps

```bash
# 1. Download landmark + 2025+ most-cited CS arXiv PDFs (~102)
python -m ingestion.download_arxiv_pdfs

# 2. Unstructured pipeline (layout + chunk → chunks.json)
python -m ingestion.unstructured_pipeline

# 3. Upload to Qdrant (pick one)
python -m ingestion.upload_qdrant_embedding --no-enrich
python -m ingestion.upload_qdrant_embedding --enrich
```

Dry-run chunk batches without API calls:

```bash
python -m ingestion.unstructured_pipeline --dry-run
```

## Outputs

| Step | Output |
|------|--------|
| 1 | `ingestion/raw_pdfs/arxiv_*.pdf`, `manifest.json` |
| 2 | `ingestion/raw_pdfs/chunks.json` |
| 3 | Qdrant collection `QDRANT_COLLECTION_NAME` (use `arxiv_cs_ds`) |

Step 3 **deletes and recreates** the collection if it already exists.

## Payload contract

Same as Citeflow corpora: embed `payload.text`; graph and RAGAS use `additional_metadata.raw_text` only. See root [README.md](../README.md) (Chunk payload contract).

arXiv `additional_metadata` includes `source` (abs URL), `page_number`, `filename`, `element_id`, `arxiv_id`. The UI shows URL and page only for cited catalog ids.

## Environment

All variables live in **`ingestion/.env`**. Key groups:

| Group | Examples |
|-------|----------|
| Qdrant + embeddings | `QDRANT_*`, `OPENAI_*`, `JINA_*`, `USE_BM25`, `USE_LATE_INTERACTION` |
| Upload | `INGESTION_UPLOAD_BATCH_SIZE` (default 16), `INGESTION_ENRICHMENT_MODEL`, `REQUEST_TIMEOUT_SECONDS` |
| Unstructured | `UNSTRUCTURED_API_KEY`, partition/chunk settings |
| Paths | `RAW_PDFS_DIR`, `CHUNKS_OUTPUT_PATH` |

Use **`arxiv_cs_ds`** as `QDRANT_COLLECTION_NAME` so you do not overwrite HotpotQA or other collections.

Full list: [`.env.example`](.env.example).

## Isolation

`ingestion/` does not import graph, retriever, API, or benchmark code.
