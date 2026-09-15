"""Always-on skip of bibliography chunks at arXiv upload (no CLI flag).

Per PDF, in Unstructured list order: enter skip on a References/Bibliography
heading, keep skipping (including tables/figures) until a new non-refs section
heading such as Appendix or Acknowledgments.
"""

from __future__ import annotations

import re
from typing import Any

_PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")

_REFS_HEADING_LINE = re.compile(
    r"""
    ^
    (?:\d+(?:\.\d+)*\s+)?
    (?:
        references?
        | bibliography
        | works\s+cited
        | literature\s+cited
        | reference\s+list
    )
    \s*[:.]?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NEW_SECTION_HEADING_LINE = re.compile(
    r"""
    ^
    (?:\d+(?:\.\d+)*\s+)?
    (?:
        appendix(?:es)?\b.*
        | supplementary\b.*
        | supplement\b.*
        | acknowledgements?\b.*
        | acknowledgments?\b.*
        | appendix\s+[a-z]\b.*
        | [a-z](?:\.\d+)*\s+appendix\b.*
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _content_lines(text: str, *, limit: int = 4) -> list[str]:
    """First non-empty lines, skipping lone page numbers."""

    found: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or _PAGE_NUMBER_LINE.fullmatch(line):
            continue
        found.append(line)
        if len(found) >= limit:
            break
    return found


def _is_refs_heading(line: str) -> bool:
    return bool(_REFS_HEADING_LINE.fullmatch(line.strip()))


def _is_new_section_heading(line: str) -> bool:
    cleaned = line.strip()
    if _is_refs_heading(cleaned):
        return False
    return bool(_NEW_SECTION_HEADING_LINE.fullmatch(cleaned))


def update_reference_skip_state(text: str, in_references: bool) -> tuple[bool, bool]:
    """Return ``(skip_this_chunk, in_references_after)``.

    Heading-only start keeps false positives low: the word ``references`` in
    body prose does not enter skip mode.
    """

    lines = _content_lines(text)
    opens_refs = any(_is_refs_heading(line) for line in lines)
    if opens_refs:
        return True, True

    if not in_references:
        return False, False

    first = lines[0] if lines else ""
    if first and _is_new_section_heading(first):
        return False, False

    return True, True


def skip_reference_chunks(elements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Filter one PDF's elements; second value is how many chunks were skipped."""

    kept: list[dict[str, Any]] = []
    skipped = 0
    in_references = False
    for element in elements:
        if not isinstance(element, dict):
            continue
        skip, in_references = update_reference_skip_state(
            str(element.get("text") or ""),
            in_references,
        )
        if skip:
            skipped += 1
            continue
        kept.append(element)
    return kept, skipped
