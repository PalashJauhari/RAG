"""Build document_catalog entries from retriever hits for graph state and prompts.

Each catalog value is keyed by Qdrant point id. ``text`` is ``payload.text`` (embed string,
including prefixed table HTML). Answer LLMs see only text, score, and id via
``catalog_for_llm``. UI/SSE cited rows may include source, page, images, and tables.
``images_base64`` is stripped before Langfuse.
"""

from __future__ import annotations

from typing import Any


def get_payload_text(payload: dict[str, Any]) -> str:
    """Return ``payload.text`` (LLM + embeddings). Fall back to ``raw_text`` if empty."""

    data = payload or {}
    text = str(data.get("text") or "")
    if text.strip():
        return text
    additional = data.get("additional_metadata")
    if isinstance(additional, dict):
        raw = additional.get("raw_text")
        if raw is not None and str(raw).strip():
            return str(raw)
    return text


def get_raw_text(payload: dict[str, Any]) -> str:
    """Alias of :func:`get_payload_text` (catalog/LLM use payload ``text``)."""

    return get_payload_text(payload)


def get_source_label(payload: dict[str, Any]) -> str:
    """Return ``additional_metadata.source`` or empty string."""

    data = payload or {}
    additional = data.get("additional_metadata")
    if isinstance(additional, dict):
        source = additional.get("source")
        if source is not None:
            return str(source).strip()
    return ""


def get_page_number(payload: dict[str, Any]) -> Any:
    """Return ``additional_metadata.page_number`` when present."""

    data = payload or {}
    additional = data.get("additional_metadata")
    if isinstance(additional, dict) and "page_number" in additional:
        return additional.get("page_number")
    return None


def _optional_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def media_from_payload(payload: dict[str, Any]) -> tuple[list[str] | None, list[str] | None]:
    """Return ``(images_base64, table_html)`` from additional_metadata; missing → None."""

    additional = (payload or {}).get("additional_metadata")
    if not isinstance(additional, dict):
        return None, None
    return (
        _optional_string_list(additional.get("images_base64")),
        _optional_string_list(additional.get("table_html")),
    )


def catalog_entries_from_retriever_hits(
    documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map retriever hits to document_catalog rows keyed by point id."""

    catalog: dict[str, dict[str, Any]] = {}
    for doc in documents:
        point_id = str(doc.get("id") or "").strip()
        if not point_id:
            continue
        payload = doc.get("payload") or {}
        images, tables = media_from_payload(payload)
        row: dict[str, Any] = {
            "text": get_payload_text(payload),
            "score": doc.get("score"),
            "source": get_source_label(payload),
            "page_number": get_page_number(payload),
            "images_base64": images,
            "table_html": tables,
        }
        catalog[point_id] = row
    return catalog


def catalog_without_images(
    catalog: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Copy catalog rows without ``images_base64`` (Langfuse / any log sink)."""

    stripped: dict[str, dict[str, Any]] = {}
    for point_id, row in (catalog or {}).items():
        if not isinstance(row, dict):
            stripped[str(point_id)] = row
            continue
        copy = dict(row)
        copy.pop("images_base64", None)
        stripped[str(point_id)] = copy
    return stripped


def cited_ui_entries(
    catalog: dict[str, dict[str, Any]] | None,
    cited_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Cited-only UI rows: source, page, images, tables. Skip missing/empty fields."""

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    catalog = catalog or {}
    for cited_id in cited_ids or []:
        key = str(cited_id or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        row = catalog.get(key)
        if not isinstance(row, dict):
            continue
        entry: dict[str, Any] = {}
        source = str(row.get("source") or "").strip()
        if source:
            entry["source"] = source
        page = row.get("page_number")
        if page is not None and str(page).strip() != "":
            entry["page_number"] = page
        images = _optional_string_list(row.get("images_base64"))
        tables = _optional_string_list(row.get("table_html"))
        if images:
            entry["images_base64"] = images
        if tables:
            entry["table_html"] = tables
        if entry:
            entries.append(entry)
    return entries


def catalog_for_client(
    catalog: dict[str, dict[str, Any]] | None,
    cited_ids: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Cited-id map for SSE/API: no passage text, no uncited rows, no empty media."""

    client: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    catalog = catalog or {}
    for cited_id in cited_ids or []:
        key = str(cited_id or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        row = catalog.get(key)
        if not isinstance(row, dict):
            continue
        entry: dict[str, Any] = {}
        source = str(row.get("source") or "").strip()
        if source:
            entry["source"] = source
        page = row.get("page_number")
        if page is not None and str(page).strip() != "":
            entry["page_number"] = page
        images = _optional_string_list(row.get("images_base64"))
        tables = _optional_string_list(row.get("table_html"))
        if images:
            entry["images_base64"] = images
        if tables:
            entry["table_html"] = tables
        if entry:
            client[key] = entry
    return client
