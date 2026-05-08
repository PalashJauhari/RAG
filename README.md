# Advanced RAG Orchestration Pipeline

This project implements a production-grade, research-backed Retrieval-Augmented Generation (RAG) orchestrator built on **LangGraph**, **Qdrant**, and **OpenAI**.

The system utilizes an agentic routing pattern that intelligently prepares queries (Expansion, Rewriting, Splitting) and executes a high-precision 3-stage retrieval pipeline to minimize hallucinations and maximize recall.

## Key Features

- **Agentic Routing**: A master orchestrator with advanced logic to choose between Precision, Recall, or Decomposition strategies.
- **Production-Grade Tools**: All preparation tools (`query_expansion`, `query_rewriter`, `query_splitter`) use Pydantic `BaseModel` schemas with strict validation, descriptive fields, and few-shot examples to guarantee schema adherence.
- **3-Stage Hybrid Retrieval**:
  1. **Stage 1 (Base Retrieval)**: Concurrent Dense (w/ MMR 3x over-fetch) and Sparse (BM25) searches.
  2. **Stage 2 (Fusion)**: Reciprocal Rank Fusion (RRF) to merge and prioritize multi-modal candidates.
  3. **Stage 3 (Re-ranking)**: Late Interaction (ColBERTv2) re-ranking for pinpoint precision.
- **Stateful Orchestration**: LangGraph-native state management with automatic conversation summarization and safe truncation on `HumanMessage` boundaries.
- **Strict Grounding**: System prompts are engineered to force grounding in retrieved context, with confidence scoring and explicit source citation.

## Architecture

```text
FastAPI /run or /resume
  -> LangGraph orchestrator (Master Planner)
  -> ToolNode (Execution Layer)
      -> query_rewriter (Context & Co-reference Resolution)
      -> query_expansion (Recall Optimization & Vocabulary Bridging)
      -> query_splitter (Multi-hop Decomposition)
      -> retrieval_tool (3-Stage Hybrid Search)
      -> ask_user (Ambiguity Resolution)
  -> orchestrator final grounded answer (answer, sources, confidence)
```

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
- `sources`: List of exact source IDs/labels used for the answer.
- `confidence`: "high", "medium", or "low".
- `retrieved_docs`: Array of documents containing `text`, `payload`, and `rank`.

## Development Notes

- **Prompts**: Centralized in the `prompts/` directory. All prompts are research-aligned with explicit "Use WHEN" triggers.
- **Tool Validation**: Tools in `tools/` use inline Pydantic models with `Field(description=..., examples=[...])` for maximum reliability with OpenAI's structured output mode.
- **LLM Client**: Unified client in `middleware/llm_client.py` handles structured output instantiation and embedding generation.
- **Observability**: Full Langfuse tracing integration enabled via `LANGFUSE_TRACING_ENABLED=true`.

## Benchmarking

HotpotQA validation and RAGAS metrics live in `benchmarking/hotpotqa`. This suite evaluates the full 3-stage pipeline against standard datasets for Precision and Recall.

---
*Powered by LangGraph, Qdrant Cloud, and Advanced RAG Research.*
