"""Data ingestion for Factline (self-contained package).

Production pipeline stages:

- :mod:`ingestion.download_data` — fetch raw files into ``ingestion/raw_pdfs/``
- :mod:`ingestion.unstructured_pipeline` — PDF partition + chunk via Unstructured Jobs API
- :mod:`ingestion.schema` — Qdrant point payload contract
- :mod:`ingestion.settings` — loads ``ingestion/.env`` only
- :mod:`ingestion.embeddings` — OpenAI dense + Jina ColBERT for upload
- :mod:`ingestion.qdrant_upload` — embed and upsert to Qdrant

See ``ingestion/README.md`` for commands and env.
"""
