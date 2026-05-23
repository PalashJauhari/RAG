"""Build prefixed embedding text from HotpotQA context records."""


def build_embedding_text(context: dict) -> str:
    """Return text for dense/BM25/ColBERT embed; raw ``text`` only when enrichment is missing."""

    passage = str(context.get("text") or "").strip()
    enrichment = context.get("enrichment")
    if not passage:
        return ""
    if not enrichment:
        return passage

    sections: list[str] = []

    title = str(context.get("title") or "").strip()
    if title:
        sections.extend(["Title:", title, ""])

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
