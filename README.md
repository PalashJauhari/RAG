<div align="center">

# Citeflow

### Evidence-first RAG for research papers and private documents

Ask a question, retrieve the supporting evidence, repair missing context, and receive a cited answer that is checked before it is returned.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## What is Citeflow?

Citeflow is a question-answering workspace for document collections where an unsupported answer is worse than an incomplete one.

Instead of retrieving a few passages and immediately asking a model to respond, Citeflow first works out what evidence the question requires. It searches for that evidence, checks whether every required fact is covered, and runs focused follow-up searches when something is missing.

The final response cites the passages it used. If the collection cannot support a complete answer within the retry budget, Citeflow returns a partial answer rather than filling the gaps with outside knowledge.

## Highlights

- **Evidence-aware answering** — answers are generated from an explicit catalog of retrieved passages.
- **Automatic retrieval repair** — uncovered facts trigger focused queries and a stronger search strategy.
- **Hybrid search** — combines dense retrieval and BM25 with reciprocal-rank fusion.
- **Optional ColBERT reranking** — improves retrieval quality when additional latency is acceptable.
- **Layout-aware PDF ingestion** — processes multi-column papers, tables, figures, and captions.
- **Cited tables and figures** — preserves rich content from supporting chunks for the chat UI.
- **Faithfulness checks** — validates citations and verifies that material answer claims are supported.
- **Safe partial answers** — stops short of a complete answer when the corpus lacks enough evidence.
- **Live progress** — streams retrieval, verification, repair, and answer events to the interface.
- **Optional observability** — traces graph nodes and model calls with Langfuse.

## Example questions

After indexing a paper collection, try:

```text
Compare the retrieval methods used by ColBERT and DPR.
Cite the passages that support the comparison.
```

```text
What assumptions do these papers make about long-context retrieval,
and where do their conclusions disagree?
```

```text
Summarize the reported benchmark results. Include the relevant table
and return a partial answer if any requested metric is unavailable.
```

## Quick start

### Prerequisites

- Python 3.10 or newer
- An OpenAI API key
- A Qdrant collection
- A Jina API key only if you want ColBERT reranking

### 1. Install

```bash
git clone https://github.com/PalashJauhari/RAG.git
cd RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI and Qdrant credentials:

```dotenv
OPENAI_API_KEY="your-key"
QDRANT_URL="https://your-cluster"
QDRANT_API_KEY="your-key"
QDRANT_COLLECTION_NAME="your-collection"
```

All settings are documented in [`.env.example`](.env.example). The local `.env` file is ignored by Git and should never be committed.

### 3. Load documents

Use the included [arXiv ingestion pipeline](ingestion/README.md) to download research papers, extract their layout, create embeddings, and upload them to Qdrant.

### 4. Run

Start the API:

```bash
uvicorn api.main:app --reload
```

In a second terminal, start the chat UI:

```bash
python -m ui.dash_app
```

Then open:

- UI: http://127.0.0.1:8050
- API: http://127.0.0.1:8000

Ask a question directly through the API:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What does the Transformer paper say about attention?"}'
```

## How it works

```text
Receive a question
    ↓
Rewrite it as a standalone query
    ↓
Identify the facts required for a complete answer
    ↓
Retrieve evidence with hybrid search
    ↓
Check whether every fact is supported
    ↓
Fill gaps and upgrade retrieval when needed
    ↓
Generate a cited full or partial answer
    ↓
Validate citations and faithfulness
```

Citeflow uses a LangGraph workflow with dedicated steps for query normalization, fact decomposition, retrieval, evidence coverage, gap filling, answer generation, and faithfulness.

The repair loop is budgeted. Each failed coverage check produces targeted queries for unsupported facts and can upgrade retrieval from Dense + BM25 to Dense + BM25 with ColBERT. The answer and partial-answer paths both pass through the same final grounding check.

## Retrieval benchmark

Retrieval was evaluated on **200 HotpotQA distractor questions**. Every configuration returned five passages. Context recall and context precision were scored with RAGAS; latency is mean retriever wall time.

| Retrieval | Quantization | Context recall | Context precision | Mean latency |
|---|---:|---:|---:|---:|
| Dense + BM25 | None | 0.82 | 0.48 | 0.51 s |
| Dense + BM25 | PQ-32 | 0.81 | 0.50 | 0.53 s |
| **Dense + BM25** | **PQ-16** | **0.80** | **0.49** | **0.52 s** |
| Dense + BM25 | PQ-8 | 0.80 | 0.48 | 0.52 s |
| Dense + BM25 + ColBERT | None | 0.86 | 0.54 | 4.90 s |
| Dense + BM25 + ColBERT | PQ-32 | 0.87 | 0.52 | 5.22 s |
| **Dense + BM25 + ColBERT** | **PQ-16** | **0.87** | **0.53** | **4.80 s** |
| Dense + BM25 + ColBERT | PQ-8 | 0.86 | 0.52 | 5.20 s |

ColBERT improves retrieval quality, with the expected latency trade-off.

## Graph benchmark

| Workflow | Quantization | Context recall | Context precision | Faithfulness | Factual correctness | Relevancy | Partial answers | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Citeflow graph** | **PQ-16** | **0.89** | **0.59** | **0.86** | **0.31** | **0.48** | **24.50%** | **28.22 s** |

The [HotpotQA benchmark guide](benchmarking/hotpotqa/README.md) contains the full setup, commands, metrics, and output format.

## Document ingestion

The ingestion pipeline is designed for research PDFs rather than plain-text documents.

It uses Unstructured for layout-aware partitioning and section-based chunking, enriches detected figures with text descriptions, preserves table structure as HTML, and stores source metadata for citations. Text, tables, and image descriptions are embedded together while original image data remains in metadata.

See [ingestion/README.md](ingestion/README.md) for setup and commands.

## Project structure

```text
RAG/
├── api/                    Application API and streaming endpoints
├── benchmarking/hotpotqa/ Offline retrieval and graph evaluation
├── config/                 Application settings
├── graph/                  LangGraph orchestration and repair loop
├── ingestion/              PDF processing, chunking, and indexing
├── middleware/             LLM clients, rate limiting, and context handling
├── observability/          Optional Langfuse tracing
├── output_validation/      Structured model outputs
├── prompts/                Instructions for graph nodes
├── retriever/              Qdrant retrieval, fusion, MMR, and reranking
├── tests/                  Test suite
└── ui/                     Plotly Dash chat interface
```

## Observability

Langfuse tracing is optional. Enable it in `.env`:

```dotenv
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

Traces cover graph nodes, retrieval decisions, model generations, retries, and final outputs.

## Development

```bash
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local workflow and contribution guidelines.

## License

Citeflow is available under the [MIT License](LICENSE).
