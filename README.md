# Advanced RAG Orchestration Pipeline

This project implements a production-grade, research-backed Retrieval-Augmented Generation (RAG) orchestrator built on **LangGraph**, **Qdrant**, and **OpenAI**.

The system uses an explicit LangGraph pipeline that plans retrieval, prepares queries, retrieves evidence, evaluates information completeness, and only then produces a grounded answer.

## Key Features

- **Typed Graph Routing**: A master orchestrator node emits validated JSON for retrieval need, decomposition, expansion, and clarification routing.
- **Evidence Evaluation Loop**: An information evaluator node checks whether the retrieved/contextual evidence is complete, then retries through the orchestrator up to a configurable limit.
- **3-Stage Hybrid Retrieval**:
  1. **Stage 1 (Base Retrieval)**: Concurrent Dense (w/ MMR 3x over-fetch) and Sparse (BM25) searches.
  2. **Stage 2 (Fusion)**: Reciprocal Rank Fusion (RRF) to merge and prioritize multi-modal candidates.
  3. **Stage 3 (Re-ranking)**: Late Interaction (ColBERTv2) re-ranking for pinpoint precision.
- **Stateful Orchestration**: LangGraph-native state management with automatic conversation summarization and safe truncation on `HumanMessage` boundaries.
- **Strict Grounding**: The answer node is prompted to answer only from message context and retrieved documents. Source citation wiring is intentionally deferred; `sources` is currently returned as an empty array.

## Architecture

```text
FastAPI /run or /resume
  -> orchestrator_node (route + rewrite)
  -> ask_user_node (optional human clarification via /resume)
  -> query_parser_node (optional decomposition first, then expansion)
  -> retrieval_node (3-Stage Hybrid Search, stored as ToolMessage)
  -> information_evaluator_node (evidence completeness check + retry routing)
  -> answer_node (answer, sources, confidence)
```

Every node appends its validated output to `state["messages"]`. Node-specific state
keys such as `orchestrator_output`, `parsed_queries`, and `information_evaluation`
are overwritten for routing convenience, but the message list remains the full audit
trail. State messages preserve provider metadata for observability; prompts are built
through plain-text formatters so response metadata is not sent back to the LLM.

## Advanced Retrieval Flow

The `Retriever` executes a sophisticated multi-stage pipeline for every query:

1. **Candidate Fetching**:
   - **Dense**: Fetches `RETRIEVAL_CANDIDATE_LIMIT * 3` raw vectors, then applies MMR (Maximal Marginal Relevance) to return a diverse pool of candidates.
   - **Sparse (BM25)**: Concurrent keyword search fetching `RETRIEVAL_CANDIDATE_LIMIT` candidates.
2. **Hybrid Fusion**:
   - Uses **Reciprocal Rank Fusion (RRF)** to merge the dense and sparse pools into a single, prioritized candidate list (top 100).
3. **Late Interaction Reranking**:
   - Uses **ColBERTv2** (via Jina multi-vectors) to perform token-level cross-attention re-ranking on the fused pool.
   - Returns the final `RETRIEVAL_TOP_K` documents.

## Environment Configuration

Copy `.env.example` to `.env`. Key performance flags:

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
INFORMATION_EVALUATION_MAX_RETRIES=5

# Node Models
ORCHESTRATOR_MODEL=gpt-5.5
INFORMATION_EVALUATOR_MODEL=gpt-5.5
FINAL_ANSWER_MODEL=gpt-5.5-mini
QUERY_DECOMPOSITION_MODEL=gpt-5.5-mini
QUERY_EXPANSION_MODEL=gpt-5.5-mini

# Observability
LANGFUSE_TRACING_ENABLED=false
```

## API Usage

Start the server:
```bash
uvicorn api.main:app --reload
```

### Run a query:
```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Compare the refund policies for Enterprise and Consumer tiers."}'
```

### Response Structure:
The system returns a structured JSON response:
- `answer`: Grounded response based strictly on context.
- `sources`: Empty for now; source extraction will be wired later.
- `confidence`: "high", "medium", or "low".
- `retrieved_docs`: Last retrieval payload, currently compacted to `score` and `text`.

When the graph needs clarification, `/run` returns:

```json
{
  "interrupted": true,
  "question": "Which policy are you asking about?",
  "answer": null
}
```

Send the user's clarification to `/resume` with the same `session_id`.

## Development Notes

- **Prompts**: Centralized in the `prompts/` directory, with separate prompts for orchestrator, query parsing, information evaluation, and final answering.
- **Output Validation**: Structured graph node outputs live in `output_validation/` and use Pydantic models with descriptive fields.
- **LLM Client**: Unified client construction in `middleware/llm_client.py` handles model configuration, structured output wrappers, rate limiting, and embedding generation.
- **Observability**: Langfuse integration is available at API, graph node, summarization, and LangChain callback layers when `LANGFUSE_TRACING_ENABLED=true`.
- **UI**: The Dash app in `ui/dash_app.py` talks to `/run` and `/resume` through `ui/api_client.py`, using the `interrupted` response flag to switch into clarification mode.

## Benchmarking

HotpotQA validation and RAGAS metrics live in `benchmarking/hotpotqa`. This suite evaluates the full 3-stage pipeline against standard datasets for Precision and Recall.

---
*Powered by LangGraph, Qdrant Cloud, and Advanced RAG Research.*
