"""Build ``graph_benchmark_report.md`` from graph benchmark JSON artifacts (internal; no CLI).

Called from :func:`~benchmarking.hotpotqa.run_graph_benchmark.run_graph_benchmark`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarking.hotpotqa.settings import HOTPOTQA_ROOT, HotpotQASettings


def load_json_file(path: Path) -> dict[str, Any] | list[Any] | None:
    """Load a JSON artifact if the path exists."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: float | int | None, digits: int = 4) -> str:
    """Format a numeric metric for Markdown tables."""

    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def relative_hotpotqa_path(path: Path) -> str:
    """Return a path relative to the HotpotQA benchmark root when possible."""

    try:
        return str(path.relative_to(HOTPOTQA_ROOT))
    except ValueError:
        return str(path)


def group_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean of ``key`` over rows where the value is not None."""

    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def build_graph_report_markdown(settings: HotpotQASettings | None = None) -> str:
    """Build graph report body and write ``graph_benchmark_report.md``.

    Args:
        settings: Benchmark settings; defaults to :class:`HotpotQASettings`.

    Returns:
        Markdown report body string.
    """
    settings = settings or HotpotQASettings()

    metadata = load_json_file(settings.graph_run_metadata_path)
    ragas = load_json_file(settings.graph_ragas_results_path)
    graph_rows = load_json_file(settings.graph_results_path)

    lines: list[str] = [
        "# HotpotQA Graph Benchmark Report",
        "",
        "## Run summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Experiment | `{settings.hotpotqa_experiment_name}` |",
        f"| Eval mode | `graph` |",
    ]

    if isinstance(metadata, dict):
        lines.extend(
            [
                f"| Questions evaluated | {metadata.get('question_count', '—')} |",
                f"| Partial answers | {metadata.get('partial_answer_count', '—')} "
                f"({format_metric(metadata.get('partial_answer_percent'), 2)}%) |",
                f"| RETRIEVAL_TOP_K | {metadata.get('retrieval_top_k', '—')} |",
                f"| RETRIEVAL_LOOP_MAX_RETRIES | {metadata.get('retrieval_loop_max_retries', '—')} |",
                f"| Timestamp (UTC) | {metadata.get('timestamp_utc', '—')} |",
            ]
        )
    elif isinstance(graph_rows, list) and graph_rows:
        partial_count = sum(1 for row in graph_rows if row.get("is_partial"))
        question_count = len(graph_rows)
        partial_pct = 100.0 * partial_count / question_count if question_count else 0.0
        lines.extend(
            [
                f"| Questions evaluated | {question_count} |",
                f"| Partial answers | {partial_count} ({format_metric(partial_pct, 2)}%) |",
            ]
        )

    lines.extend(
        [
            "",
            "## RAGAS (context, faithfulness, answer correctness)",
            "",
            "_Partial-answer rows are included in means when scorable (empty response or contexts skipped per metric)._",
            "",
        ]
    )
    if isinstance(ragas, dict) and ragas.get("summary"):
        s = ragas["summary"]
        lines.extend(
            [
                "| Metric | Mean |",
                "|--------|------|",
                f"| Context precision | {format_metric(s.get('mean_context_precision'))} |",
                f"| Context recall | {format_metric(s.get('mean_context_recall'))} |",
                f"| Faithfulness | {format_metric(s.get('mean_faithfulness'))} |",
                f"| Answer correctness | {format_metric(s.get('mean_answer_correctness'))} |",
                f"| Questions scored | {s.get('total_scored', '—')} |",
                f"| Skipped (empty contexts) | {s.get('skipped_empty_contexts', 0)} |",
                f"| Skipped (empty response) | {s.get('skipped_empty_response', 0)} |",
                f"| Partial answers (input set) | {s.get('partial_answer_count', '—')} |",
                "",
            ]
        )
    else:
        lines.append("_No `graph_ragas_results.json` found._\n")

    lines.extend(["", "## Graph latency", ""])
    lines.append(
        "_Wall-clock time for `RetrievalGraph.run()` per question only — not RAGAS scoring._\n"
    )
    latency: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        latency = metadata.get("graph_latency")
    if latency:
        lines.extend(
            [
                "| Stat | ms |",
                "|------|-----|",
                f"| Mean | {format_metric(latency.get('mean_ms'), 2)} |",
                f"| p50 | {format_metric(latency.get('p50_ms'), 2)} |",
                f"| p95 | {format_metric(latency.get('p95_ms'), 2)} |",
                f"| Min | {format_metric(latency.get('min_ms'), 2)} |",
                f"| Max | {format_metric(latency.get('max_ms'), 2)} |",
                "",
            ]
        )
    elif isinstance(graph_rows, list) and graph_rows:
        latencies = [
            float(row["graph_latency_ms"])
            for row in graph_rows
            if row.get("graph_latency_ms") is not None
        ]
        if latencies:
            ordered = sorted(latencies)
            n = len(ordered)
            lines.extend(
                [
                    "| Stat | ms |",
                    "|------|-----|",
                    f"| Mean | {format_metric(sum(ordered) / n, 2)} |",
                    f"| Min | {format_metric(ordered[0], 2)} |",
                    f"| Max | {format_metric(ordered[-1], 2)} |",
                    "",
                ]
            )
    else:
        lines.append("_No latency data found._\n")

    if isinstance(ragas, dict) and ragas.get("rows"):
        lines.extend(["", "## Breakdown by type and level (RAGAS means)", ""])
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in ragas["rows"]:
            groups[(row.get("type", "unknown"), row.get("level", "unknown"))].append(row)

        lines.extend(
            [
                "| Type | Level | Count | Context precision | Context recall | Faithfulness | Answer correctness |",
                "|------|-------|-------|-------------------|----------------|--------------|-------------------|",
            ]
        )
        for (qtype, level), group_rows in sorted(groups.items()):
            n = len(group_rows)
            lines.append(
                f"| {qtype} | {level} | {n} | "
                f"{format_metric(group_mean(group_rows, 'context_precision'))} | "
                f"{format_metric(group_mean(group_rows, 'context_recall'))} | "
                f"{format_metric(group_mean(group_rows, 'faithfulness'))} | "
                f"{format_metric(group_mean(group_rows, 'answer_correctness'))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{relative_hotpotqa_path(settings.graph_results_path)}`",
            f"- `{relative_hotpotqa_path(settings.graph_run_metadata_path)}`",
            f"- `{relative_hotpotqa_path(settings.graph_ragas_results_path)}`",
            f"- `{relative_hotpotqa_path(settings.graph_benchmark_report_path)}`",
            "",
        ]
    )

    body = "\n".join(lines)
    settings.graph_benchmark_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.graph_benchmark_report_path.write_text(body, encoding="utf-8")
    print(f"Wrote graph benchmark report to {settings.graph_benchmark_report_path}")
    return body
