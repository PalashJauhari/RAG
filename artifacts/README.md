# Artifacts

Generated files for documentation (not committed — see root `.gitignore`).

## LangGraph topology

| File | Description |
|------|-------------|
| `langgraph.mmd` | Mermaid source from the compiled `RetrievalGraph` |
| `langgraph.png` | Rendered diagram (linked from root `README.md`) |

Regenerate from repo root with **`rag_env_1`** active:

```bash
source /path/to/rag_env_1/bin/activate
python scripts/plot_langgraph.py
```

If Mermaid.ink fails, use a local browser renderer:

```bash
python scripts/plot_langgraph.py --draw-method pyppeteer
```

Benchmark result reports live under `benchmarking/hotpotqa/data/results/{experiment}/` (also gitignored), not in this folder.
