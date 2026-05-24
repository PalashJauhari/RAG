"""Aggregate benchmark JSON artifacts into ``benchmark_report.md``.

Run: ``python -m benchmarking.hotpotqa.metrics.build_report``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarking.hotpotqa.settings import HOTPOTQA_ROOT, HotpotQASettings


def load_json_file(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def relative_hotpotqa_path(path: Path) -> str:
    try:
        return str(path.relative_to(HOTPOTQA_ROOT))
    except ValueError:
        return str(path)


def build_report_markdown(settings: HotpotQASettings | None = None) -> str:
    """Build report body and write ``benchmark_report.md``."""

    settings = settings or HotpotQASettings()

    metadata = load_json_file(settings.run_metadata_path)
    ragas = load_json_file(settings.ragas_results_path)
    retrieval_rows = load_json_file(settings.retrieval_results_path)

    strategy = None
    if isinstance(metadata, dict):
        strategy = metadata.get("retrieval_strategy")
    elif isinstance(retrieval_rows, list) and retrieval_rows:
        strategy = retrieval_rows[0].get("retrieval_strategy")

    lines: list[str] = [
        "# HotpotQA Retriever Benchmark Report",
        "",
        "## Run summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Experiment | `{settings.hotpotqa_experiment_name}` |",
        f"| Retrieval strategy | `{strategy or 'unknown'}` |",
    ]

    if isinstance(metadata, dict):
        lines.extend(
            [
                f"| Questions evaluated | {metadata.get('question_count', '—')} |",
                f"| RETRIEVAL_TOP_K | {metadata.get('retrieval_top_k', '—')} |",
                f"| Timestamp (UTC) | {metadata.get('timestamp_utc', '—')} |",
            ]
        )

    lines.extend(["", "## RAGAS (context precision & recall)", ""])
    if isinstance(ragas, dict) and ragas.get("summary"):
        s = ragas["summary"]
        lines.extend(
            [
                "| Metric | Mean |",
                "|--------|------|",
                f"| Context precision | {format_metric(s.get('mean_context_precision'))} |",
                f"| Context recall | {format_metric(s.get('mean_context_recall'))} |",
                f"| Questions scored | {s.get('total_scored', '—')} |",
                f"| Skipped (empty contexts) | {s.get('skipped_empty_contexts', 0)} |",
                "",
            ]
        )
    else:
        lines.append("_No `ragas_results.json` found (run without `--skip-ragas`?)._\n")

    lines.extend(["", "## Retriever latency", ""])
    lines.append(
        "_Wall-clock time for `Retriever.retrieve()` per question only — not RAGAS scoring._\n"
    )
    latency: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        latency = metadata.get("retriever_latency")
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
    elif isinstance(retrieval_rows, list) and retrieval_rows:
        latencies = [
            float(r["retrieval_latency_ms"])
            for r in retrieval_rows
            if r.get("retrieval_latency_ms") is not None
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
        from collections import defaultdict

        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in ragas["rows"]:
            groups[(row.get("type", "unknown"), row.get("level", "unknown"))].append(row)

        lines.extend(
            [
                "| Type | Level | Count | Context precision | Context recall |",
                "|------|-------|-------|-------------------|----------------|",
            ]
        )
        for (qtype, level), group_rows in sorted(groups.items()):
            n = len(group_rows)
            lines.append(
                f"| {qtype} | {level} | {n} | "
                f"{format_metric(sum(r['context_precision'] for r in group_rows) / n)} | "
                f"{format_metric(sum(r['context_recall'] for r in group_rows) / n)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{relative_hotpotqa_path(settings.retrieval_results_path)}`",
            f"- `{relative_hotpotqa_path(settings.run_metadata_path)}`",
            f"- `{relative_hotpotqa_path(settings.ragas_results_path)}`",
            f"- `{relative_hotpotqa_path(settings.benchmark_report_path)}`",
            "",
        ]
    )

    body = "\n".join(lines)
    settings.benchmark_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.benchmark_report_path.write_text(body, encoding="utf-8")
    print(f"Wrote benchmark report to {settings.benchmark_report_path}")
    return body


def main() -> None:
    build_report_markdown()


if __name__ == "__main__":
    main()
