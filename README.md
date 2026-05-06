# RAG Retrieval Orchestrator

This project implements the retrieval and answer orchestration layer for a RAG system.

The graph prepares the user query, retrieves documents from Qdrant Cloud, and generates a final JSON answer grounded in the retrieved context.

## Architecture

```text
FastAPI /run or /resume
  -> LangGraph orchestrator
  -> ToolNode
      -> query_rewriter
      -> query_expansion
      -> query_splitter
      -> retrieval_tool
      -> ask_user
  -> orchestrator final grounded answer
```

The graph uses `session_id` as the LangGraph `thread_id`, so parallel sessions stay isolated. Checkpoints use `InMemorySaver` by default, or Postgres when `CHECKPOINTER_USE_POSTGRES=true`.

## Retrieval Flow

```text
prepared queries
  -> OpenAI dense embeddings
  -> optional Qdrant BM25 candidate retrieval
  -> optional Qdrant hybrid fusion
  -> optional Qdrant MMR on dense candidate retrieval
  -> optional Jina ColBERTv2 query multivectors
  -> optional Qdrant late-interaction rerank
  -> final top-k documents
```

Qdrant-native behavior is used for BM25 query inference, hybrid fusion, MMR, and multivector late interaction. Jina is used only to generate the ColBERTv2 query multivectors.

## Qdrant Collection Contract

The Qdrant collection must already be populated. Vector names are configured in `.env`.

Required when enabled:

- `QDRANT_DENSE_VECTOR_NAME`: dense semantic vectors matching `OPENAI_EMBEDDING_MODEL`
- `QDRANT_BM25_VECTOR_NAME`: BM25 sparse vectors, required when `USE_BM25=true`
- `QDRANT_COLBERT_VECTOR_NAME`: Jina ColBERTv2 multivectors, required when `USE_LATE_INTERACTION=true`

The ColBERT vector should use Qdrant `MAX_SIM` multivector comparison and the same dimension as `JINA_COLBERT_DIMENSIONS`.

## Environment

Copy `.env.example` to `.env` and fill values.

Important flags:

```env
USE_BM25=true
USE_LATE_INTERACTION=true
USE_MMR=true

QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_BM25_VECTOR_NAME=bm25
QDRANT_COLBERT_VECTOR_NAME=colbert
```

These flags are loaded into the retriever config and used when creating the Qdrant client. `USE_BM25=true` enables Qdrant cloud inference on the client.

Harness flags:

```env
CHECKPOINTER_USE_POSTGRES=false
DATABASE_URL=

GRAPH_RECURSION_LIMIT=100
GRAPH_MAX_CONCURRENCY=2

MESSAGE_SUMMARY_TOKEN_THRESHOLD=100000
MESSAGE_SUMMARY_KEEP_RECENT=10

OPENAI_RATE_LIMIT_ENABLED=true
OPENAI_RATE_LIMIT_REQUESTS_PER_SECOND=1.0
```

When Postgres checkpoints are enabled, install `langgraph-checkpoint-postgres` and `psycopg[binary]`, set `DATABASE_URL`, and the API startup will run `checkpointer.setup()`.

## API

Start the API:

```bash
uvicorn api.main:app --reload
```

Run a new turn:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What does the policy say about refunds?","recursion_limit":25}'
```

Resume after `ask_user` interrupts:

```bash
curl -X POST http://127.0.0.1:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","answer":"Use the enterprise customer policy","recursion_limit":25}'
```

Response shape:

```json
{
  "session_id": "demo",
  "interrupted": false,
  "question": null,
  "answer": "...",
  "sources": [],
  "confidence": "medium",
  "retrieved_docs": []
}
```

Interrupt shape:

```json
{
  "session_id": "demo",
  "interrupted": true,
  "question": "Which policy should I use?",
  "answer": null,
  "sources": [],
  "confidence": null,
  "retrieved_docs": []
}
```

## Dash UI

Start the API first, then run:

```bash
python ui/dash_app.py
```

Open `http://127.0.0.1:8050`.

## Notes

- Tool LLM calls use OpenAI Structured Outputs with Pydantic schemas in `output_validation/`.
- `graph/RetrievalGraph` wraps graph construction, run, resume, state lookup, and checkpoint setup.
- `middleware/` contains the shared LLM client, optional OpenAI rate limiter, and context summarization.
- `ui/api_client.py` centralizes Dash-to-FastAPI HTTP calls.
- The orchestrator is instructed to return final JSON with `answer`, `sources`, and `confidence`.
- Langfuse tracing is controlled by `LANGFUSE_TRACING_ENABLED`. Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` when enabled.

## Benchmarking

HotpotQA local retrieval benchmarking lives in `benchmarking/hotpotqa`.

It uses a separate `.env`, prepares HotpotQA fullwiki validation data from Hugging Face, uploads paragraph contexts to Qdrant, evaluates the `Retriever` directly, and runs RAGAS context precision/recall metrics.

## References

- Qdrant hybrid queries: https://qdrant.tech/documentation/search/hybrid-queries/
- Qdrant BM25 cloud inference: https://qdrant.tech/documentation/inference/
- Qdrant MMR: https://qdrant.tech/documentation/search/search-relevance/
- Jina ColBERTv2 multivectors: https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangChain ToolRuntime and ToolNode: https://docs.langchain.com/oss/python/langchain/tools
- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Langfuse LangGraph integration: https://langfuse.com/guides/cookbook/integration_langgraph
