# Artifacts

Generated files for documentation (not committed — see root `.gitignore`).

The README flowchart is Mermaid in `README.md` (happy path + repair loop; no node-failure handlers).

Optional PNG/MMD dump from the compiled graph (error-handler nodes stripped from `.mmd`):

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
