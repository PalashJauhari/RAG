"""Download arXiv PDFs into ``ingestion/raw_pdfs/`` and write ``manifest.json``.

Default corpus: two landmark papers plus the most-cited 2025+ OpenAlex matches
in five CS slices (time series, LLMs, CV, ML, agents). About 102 PDFs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

from ingestion.ingestion_config import INGESTION_ROOT, load_ingestion_config

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"

REQUEST_TIMEOUT_SECONDS = 60
OPENALEX_DELAY_SECONDS = 0.2
ARXIV_DELAY_SECONDS = 3.0
OPENALEX_PER_PAGE = 50
OPENALEX_MAX_PAGES = 8

ARXIV_ID_RE = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/|:))?"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:\.pdf)?",
    re.IGNORECASE,
)

PINNED_PAPERS: list[dict[str, str]] = [
    {
        "arxiv_id": "1706.03762",
        "slice": "landmark",
        "title": "Attention Is All You Need",
    },
    {
        "arxiv_id": "2210.03629",
        "slice": "landmark",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
    },
]

# OpenAlex search query + how many PDFs to take (after landmark pins).
SLICES: list[tuple[str, str, int]] = [
    ("time_series", "time series forecasting", 20),
    ("llms", "large language models", 20),
    ("cv", "computer vision", 20),
    ("ml", "machine learning", 20),
    ("agents", "large language model agents", 20),
]


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return a versionless arXiv id, or None if the string is not an id."""

    text = str(value or "").strip()
    if not text:
        return None
    match = ARXIV_ID_RE.search(text)
    if not match:
        return None
    arxiv_id = match.group(1)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)
    return arxiv_id or None


def abs_url_for(arxiv_id: str) -> str:
    """Canonical abstract URL stored as Qdrant ``additional_metadata.source``."""

    return ARXIV_ABS_URL.format(arxiv_id=arxiv_id)


def pdf_url_for(arxiv_id: str) -> str:
    """arXiv PDF download URL."""

    return ARXIV_PDF_URL.format(arxiv_id=arxiv_id)


def pdf_filename_for(arxiv_id: str) -> str:
    """Filesystem-safe PDF name."""

    return f"arxiv_{arxiv_id.replace('/', '_')}.pdf"


def extract_arxiv_id_from_work(work: dict[str, Any]) -> str | None:
    """Pull an arXiv id from an OpenAlex work payload."""

    ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
    found = normalize_arxiv_id(str(ids.get("arxiv") or ""))
    if found:
        return found

    for location in work.get("locations") or []:
        if not isinstance(location, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            found = normalize_arxiv_id(str(location.get(key) or ""))
            if found:
                return found
        source = location.get("source") if isinstance(location.get("source"), dict) else {}
        display = str(source.get("display_name") or "")
        if "arxiv" in display.lower():
            found = normalize_arxiv_id(str(location.get("landing_page_url") or ""))
            if found:
                return found
    return None


def clear_output_dir(output_dir: Path) -> None:
    """Delete all contents of ``output_dir`` except ``.gitkeep``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    print(f"Cleared {output_dir}/")


def relative_pdf_path(pdf_path: Path) -> str:
    """Return ``pdf_path`` relative to ingestion root when possible."""

    try:
        return str(pdf_path.relative_to(INGESTION_ROOT))
    except ValueError:
        return str(pdf_path)


def request_headers() -> dict[str, str]:
    """User-Agent for OpenAlex and arXiv (include mailto when set)."""

    email = os.getenv("CONTACT_EMAIL", "").strip()
    ua = "FactlineRAG/1.0 (https://github.com/arxiv/arxiv-docs)"
    if email:
        ua = f"FactlineRAG/1.0 (mailto:{email})"
    return {"User-Agent": ua}


def openalex_params(search: str, cursor: str, mailto: str) -> dict[str, str]:
    """Query params for one OpenAlex page: 2025+ works, citation desc."""

    params = {
        "search": search,
        "filter": (
            "from_publication_date:2025-01-01,"
            "primary_location.source.id:https://openalex.org/S4306400194"
        ),
        "sort": "cited_by_count:desc",
        "per_page": str(OPENALEX_PER_PAGE),
        "cursor": cursor,
        "select": "id,display_name,publication_year,cited_by_count,ids,locations,open_access",
    }
    if mailto:
        params["mailto"] = mailto
    return params


def search_openalex_slice(
    *,
    slice_name: str,
    search: str,
    target: int,
    seen_ids: set[str],
    mailto: str,
) -> list[dict[str, Any]]:
    """Return up to ``target`` unique arXiv papers for one topical slice."""

    selected: list[dict[str, Any]] = []
    cursor = "*"
    session = requests.Session()
    session.headers.update(request_headers())

    for _page in range(OPENALEX_MAX_PAGES):
        if len(selected) >= target:
            break
        response = session.get(
            OPENALEX_URL,
            params=openalex_params(search, cursor, mailto),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        for work in payload.get("results") or []:
            if not isinstance(work, dict):
                continue
            arxiv_id = extract_arxiv_id_from_work(work)
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            selected.append(
                {
                    "arxiv_id": arxiv_id,
                    "slice": slice_name,
                    "title": str(work.get("display_name") or "").strip() or None,
                    "cited_by_count": work.get("cited_by_count"),
                    "publication_year": work.get("publication_year"),
                    "openalex_id": work.get("id"),
                }
            )
            if len(selected) >= target:
                break
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        cursor = str(meta.get("next_cursor") or "")
        if not cursor:
            break
        time.sleep(OPENALEX_DELAY_SECONDS)

    print(f"Slice {slice_name!r}: selected {len(selected)} arXiv id(s) (target {target}).")
    return selected


def download_arxiv_pdf(arxiv_id: str) -> bytes:
    """Fetch PDF bytes from arXiv."""

    response = requests.get(
        pdf_url_for(arxiv_id),
        headers=request_headers(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("Downloaded payload is not a PDF.")
    return response.content


def paper_row(
    *,
    arxiv_id: str,
    slice_name: str,
    title: str | None,
    abs_url: str,
    pdf_url: str,
    pdf_path: Path | None,
    status: str,
    cited_by_count: Any = None,
    publication_year: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one manifest entry."""

    row: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "slice": slice_name,
        "title": title,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "pdf_path": relative_pdf_path(pdf_path) if pdf_path else None,
        "cited_by_count": cited_by_count,
        "publication_year": publication_year,
        "status": status,
    }
    if error:
        row["error"] = error
    return row


def collect_paper_specs(mailto: str) -> list[dict[str, Any]]:
    """Pinned landmarks first, then OpenAlex slices."""

    seen: set[str] = set()
    specs: list[dict[str, Any]] = []
    for pin in PINNED_PAPERS:
        arxiv_id = pin["arxiv_id"]
        seen.add(arxiv_id)
        specs.append(
            {
                "arxiv_id": arxiv_id,
                "slice": pin["slice"],
                "title": pin["title"],
                "cited_by_count": None,
                "publication_year": None,
            }
        )

    for slice_name, search, target in SLICES:
        specs.extend(
            search_openalex_slice(
                slice_name=slice_name,
                search=search,
                target=target,
                seen_ids=seen,
                mailto=mailto,
            )
        )
    return specs


def run_download(*, output_dir: Path) -> dict[str, Any]:
    """Clear output dir, resolve paper list, download PDFs, return manifest."""

    clear_output_dir(output_dir)
    mailto = os.getenv("CONTACT_EMAIL", "").strip()
    print("Selecting arXiv papers (landmarks + OpenAlex 2025+ citation rank)...")
    specs = collect_paper_specs(mailto)
    print(f"Will attempt {len(specs)} downloads.")

    papers: list[dict[str, Any]] = []
    downloaded_count = 0
    failed_count = 0

    for index, spec in enumerate(specs, start=1):
        arxiv_id = spec["arxiv_id"]
        title = spec.get("title")
        title_preview = (title or arxiv_id)[:60]
        filepath = output_dir / pdf_filename_for(arxiv_id)
        print(f"[{index}/{len(specs)}] {title_preview}...")
        try:
            pdf_bytes = download_arxiv_pdf(arxiv_id)
            filepath.write_bytes(pdf_bytes)
            papers.append(
                paper_row(
                    arxiv_id=arxiv_id,
                    slice_name=str(spec.get("slice") or ""),
                    title=title,
                    abs_url=abs_url_for(arxiv_id),
                    pdf_url=pdf_url_for(arxiv_id),
                    pdf_path=filepath,
                    status="downloaded",
                    cited_by_count=spec.get("cited_by_count"),
                    publication_year=spec.get("publication_year"),
                )
            )
            downloaded_count += 1
            print(f"  -> saved {filepath.name} [{downloaded_count}/{len(specs)}]")
        except (OSError, requests.RequestException, ValueError) as exc:
            failed_count += 1
            papers.append(
                paper_row(
                    arxiv_id=arxiv_id,
                    slice_name=str(spec.get("slice") or ""),
                    title=title,
                    abs_url=abs_url_for(arxiv_id),
                    pdf_url=pdf_url_for(arxiv_id),
                    pdf_path=None,
                    status="failed",
                    cited_by_count=spec.get("cited_by_count"),
                    publication_year=spec.get("publication_year"),
                    error=str(exc),
                )
            )
            print(f"  -> failed: {exc}")
        time.sleep(ARXIV_DELAY_SECONDS)

    manifest = {
        "source": "arxiv",
        "collection_name": "arxiv_cs_ds",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_count": downloaded_count,
        "failed_count": failed_count,
        "candidate_count": len(specs),
        "papers": papers,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Done. {downloaded_count} downloaded, {failed_count} failed, "
        f"{len(specs)} candidates. Manifest: {manifest_path}"
    )
    return manifest


def main() -> None:
    """CLI entry for arXiv PDF download."""

    parser = argparse.ArgumentParser(
        description=(
            "Download landmark + 2025+ most-cited CS arXiv PDFs into ingestion/raw_pdfs/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ingestion/raw_pdfs).",
    )
    args = parser.parse_args()
    config = load_ingestion_config()
    output_dir = args.output_dir or config.resolve_path(config.raw_pdfs_dir)
    if not output_dir.is_absolute():
        output_dir = INGESTION_ROOT / output_dir
    run_download(output_dir=output_dir)


if __name__ == "__main__":
    main()
