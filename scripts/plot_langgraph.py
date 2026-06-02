"""Render the compiled LangGraph to Mermaid source and PNG (topology only).

Default outputs: ``artifacts/langgraph.mmd`` and ``artifacts/langgraph.png``.
Does **not** generate ``langgraph_io.*`` state I/O diagrams — those are intentionally
unsupported. Uses in-memory checkpointing only (no Postgres required).

Example::

    python scripts/plot_langgraph.py --draw-method pyppeteer
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph import RetrievalGraph


def _reject_io_artifact_path(path: Path, label: str) -> None:
    """Topology script must not write ``*_io`` artifact names."""
    if path.stem.endswith("_io"):
        raise SystemExit(
            f"{label} must not use *_io names (topology only): {path}. "
            "Use artifacts/langgraph.mmd and artifacts/langgraph.png."
        )


def _remove_stale_io_artifacts(artifacts_dir: Path) -> None:
    """Delete legacy hand-maintained I/O diagram files if they still exist."""
    for pattern in ("langgraph_io.mmd", "langgraph_io.png", "*_io.mmd", "*_io.png"):
        for stale in artifacts_dir.glob(pattern):
            stale.unlink()
            print(f"Removed stale artifact {stale}")


async def main() -> None:
    """Compile ``RetrievalGraph``, write Mermaid text, and render PNG via Mermaid.ink or Pyppeteer."""
    parser = argparse.ArgumentParser(
        description="Render the compiled LangGraph RAG flow to Mermaid and PNG files.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/langgraph.png",
        help="PNG path to write. Defaults to artifacts/langgraph.png.",
    )
    parser.add_argument(
        "--mermaid-output",
        default="artifacts/langgraph.mmd",
        help="Mermaid source path to write. Defaults to artifacts/langgraph.mmd.",
    )
    parser.add_argument(
        "--draw-method",
        choices=[method.value for method in MermaidDrawMethod],
        default=MermaidDrawMethod.API.value,
        help="PNG renderer: api uses Mermaid.ink; pyppeteer uses a local browser.",
    )
    parser.add_argument(
        "--background-color",
        default="white",
        help="PNG background color.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="PNG padding in pixels.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    mermaid_path = Path(args.mermaid_output)
    _reject_io_artifact_path(output_path, "--output")
    _reject_io_artifact_path(mermaid_path, "--mermaid-output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_io_artifacts(output_path.parent)

    retrieval_graph = RetrievalGraph(InMemorySaver())
    try:
        graph = retrieval_graph.graph.get_graph()
        mermaid = graph.draw_mermaid()
        mermaid_path.write_text(mermaid, encoding="utf-8")

        png = graph.draw_mermaid_png(
            output_file_path=str(output_path),
            draw_method=MermaidDrawMethod(args.draw_method),
            background_color=args.background_color,
            padding=args.padding,
            max_retries=3,
            retry_delay=1.0,
        )
        if not output_path.exists():
            output_path.write_bytes(png)
    finally:
        await retrieval_graph.retriever.qdrant.close()

    print(f"Wrote Mermaid graph to {mermaid_path}")
    print(f"Wrote graph image to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
