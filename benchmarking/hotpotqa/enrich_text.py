"""Build enriched search strings from raw passages and optional enrichments."""

from __future__ import annotations

from typing import Any


def build_enriched_text(
    *,
    raw_text: str,
    enrichments: dict[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Return enriched text for dense/BM25/ColBERT embedding at upload.

    When ``enrichments`` is empty, returns ``raw_text`` unchanged.
    """

    passage = str(raw_text or "").strip()
    if not passage:
        return ""

    enrichment = enrichments or {}
    if not enrichment:
        return passage

    sections: list[str] = []

    title_text = str(title or "").strip()
    if title_text:
        sections.extend(["Title:", title_text, ""])

    summary = str(enrichment.get("summary") or "").strip()
    if summary:
        sections.extend(["Summary:", summary, ""])

    facts = enrichment.get("facts") or []
    fact_lines = [
        f"- {str(row.get('fact') or '').strip()}"
        for row in facts
        if isinstance(row, dict) and str(row.get("fact") or "").strip()
    ]
    if fact_lines:
        sections.append("Present Facts:")
        sections.extend(fact_lines)
        sections.append("")

    question_lines = [
        f"- {str(row.get('fact_question') or '').strip()}"
        for row in facts
        if isinstance(row, dict) and str(row.get("fact_question") or "").strip()
    ]
    if question_lines:
        sections.append("Sample Query Questions:")
        sections.extend(question_lines)
        sections.append("")

    keywords = enrichment.get("keywords") or []
    keyword_text = ", ".join(
        str(item).strip() for item in keywords if isinstance(item, str) and item.strip()
    )
    if keyword_text:
        sections.extend(["Keywords:", keyword_text, ""])

    sections.extend(["Passage:", passage])
    return "\n".join(sections)
