"""Extract image bytes and table HTML from Unstructured chunk elements."""

from __future__ import annotations

from typing import Any

TABLE_HTML_PREFIX = "Relevant table data:"


def nonempty_string_list(value: Any) -> list[str] | None:
    """Return stripped strings, or None when missing/empty."""

    if not isinstance(value, list) or not value:
        return None
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def extract_images_base64(element: dict[str, Any]) -> list[str] | None:
    """Collect image base64 strings from chunk metadata or orig_elements."""

    metadata = element.get("metadata")
    if not isinstance(metadata, dict):
        return None

    images = metadata.get("images")
    collected: list[str] = []
    if isinstance(images, list):
        for item in images:
            if isinstance(item, str) and item.strip():
                collected.append(item.strip())
            elif isinstance(item, dict):
                data = str(item.get("base64") or "").strip()
                if data:
                    collected.append(data)

    orig = metadata.get("orig_elements")
    if isinstance(orig, list):
        for item in orig:
            if not isinstance(item, dict) or item.get("type") != "Image":
                continue
            nested = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            data = str(nested.get("image_base64") or "").strip()
            if data and data not in collected:
                collected.append(data)

    return collected or None


def extract_table_html(element: dict[str, Any]) -> list[str] | None:
    """Collect ``text_as_html`` from Table orig_elements."""

    metadata = element.get("metadata")
    if not isinstance(metadata, dict):
        return None
    orig = metadata.get("orig_elements")
    if not isinstance(orig, list):
        return None

    tables: list[str] = []
    for item in orig:
        if not isinstance(item, dict) or item.get("type") != "Table":
            continue
        nested = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        html = str(nested.get("text_as_html") or "").strip()
        if html:
            tables.append(html)
    return tables or None


def with_prefixed_tables(text: str, tables: list[str] | None) -> str:
    """Append each HTML table with ``Relevant table data:`` to embed/LLM ``text``."""

    base = str(text or "").strip()
    if not tables:
        return base
    parts: list[str] = [base] if base else []
    for html in tables:
        block = str(html or "").strip()
        if not block:
            continue
        parts.append(f"{TABLE_HTML_PREFIX}\n{block}")
    return "\n\n".join(parts)
