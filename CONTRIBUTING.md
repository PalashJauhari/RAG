# Contributing

Thanks for helping improve Factline / RAG.

## Prerequisites

- Python **3.10+** (3.12 recommended)
- A virtualenv and `pip install -r requirements.txt`
- Root `.env` copied from `.env.example` (at least `OPENAI_API_KEY` and Qdrant settings for full app runs)

## Local workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test          # or: pytest
./start.sh         # API :8000 + UI :8050
./stop.sh          # stop both
```

## Pull requests

1. Keep changes focused; match existing style and layout.
2. Add or update tests under `tests/` when behavior changes.
3. Do not commit secrets, `.env`, or local data under `ingestion/raw_pdfs/`, benchmark raw/processed dumps, or regenerated `artifacts/` except documented topology images.
4. Prefer clear commit messages that explain **why**.

## Where to change things

| Area | Location |
|------|----------|
| Main graph | `graph/graph.py` |
| Retriever | `retriever/` |
| API | `api/main.py` |
| UI | `ui/` |
| Prompts | `prompts/` |
| Output schemas | `output_validation/` |
| Ingestion | `ingestion/` |
| Benchmarks | `benchmarking/hotpotqa/` |

## License

By contributing, you agree your contributions are licensed under the MIT License (see `LICENSE`).
