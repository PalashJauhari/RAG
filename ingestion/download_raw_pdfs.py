"""Download open-access PMC PDFs into ``ingestion/raw_pdfs/`` and write ``manifest.json``."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
import time
import zlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from ftplib import FTP
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ingestion.ingestion_config import INGESTION_ROOT

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_FTP_HOST = "ftp.ncbi.nlm.nih.gov"

REQUEST_TIMEOUT_SECONDS = 60
PMC_REQUEST_DELAY_SECONDS = 0.4
NCBI_TOOL = "factline-rag"
SEARCH_OVERFETCH_FACTOR = 3


def validate_pmc_query(query: str) -> str:
    """Normalize and validate a required PMC Entrez ``--query`` value."""

    key = (query or "").strip()
    if not key:
        raise ValueError("--query is required and must be non-empty")
    return key


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


def ncbi_params(**extra: str | int) -> dict[str, str | int]:
    """Base query params recommended by NCBI E-utilities."""

    params: dict[str, str | int] = {"tool": NCBI_TOOL}
    email = os.getenv("NCBI_EMAIL", "").strip()
    if email:
        params["email"] = email
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    params.update(extra)
    return params


def search_pmc_ids(query: str, retmax: int) -> list[str]:
    """Return numeric PMC ids from an Entrez ``esearch`` query."""

    response = requests.get(
        ESEARCH_URL,
        params=ncbi_params(db="pmc", term=query, retmax=retmax, retmode="json"),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("esearchresult", {}).get("idlist", []))


def fetch_pmc_titles(pmc_ids: list[str]) -> dict[str, str]:
    """Batch-fetch titles for PMC ids via ``esummary``."""

    if not pmc_ids:
        return {}

    titles: dict[str, str] = {}
    chunk_size = 200
    for start in range(0, len(pmc_ids), chunk_size):
        chunk = pmc_ids[start : start + chunk_size]
        response = requests.get(
            ESUMMARY_URL,
            params=ncbi_params(db="pmc", id=",".join(chunk), retmode="json"),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        for pmc_id in chunk:
            record = result.get(pmc_id)
            if isinstance(record, dict):
                title = str(record.get("title") or "").strip()
                if title:
                    titles[pmc_id] = title
        time.sleep(PMC_REQUEST_DELAY_SECONDS)
    return titles


def normalize_ftp_path(href: str) -> str:
    """Rewrite legacy PMC OA FTP paths to the ``deprecated`` directory."""

    href = href.strip()
    if href.startswith("/ftp://"):
        href = href.lstrip("/")
    parsed = urlparse(href)
    if parsed.scheme != "ftp" or parsed.netloc != PMC_FTP_HOST:
        return href

    path = parsed.path
    legacy_prefixes = ("/pub/pmc/oa_pdf/", "/pub/pmc/oa_package/")
    for prefix in legacy_prefixes:
        if path.startswith(prefix):
            return f"ftp://{PMC_FTP_HOST}/pub/pmc/deprecated/{path.removeprefix('/pub/pmc/')}"
    return href


def ftp_fetch_bytes(ftp_path: str) -> bytes:
    """Download a file from NCBI PMC FTP via ``RETR``."""

    parsed = urlparse(ftp_path)
    if parsed.scheme != "ftp":
        raise ValueError(f"Not an FTP URL: {ftp_path}")

    remote_path = parsed.path
    buffer = io.BytesIO()
    ftp = FTP(PMC_FTP_HOST, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        ftp.login()
        ftp.retrbinary(f"RETR {remote_path}", buffer.write)
    finally:
        try:
            ftp.quit()
        except OSError:
            pass
    return buffer.getvalue()


def extract_first_pdf_from_tgz(data: bytes) -> bytes:
    """Return the first PDF member from a PMC OA ``tgz`` package."""

    if not data:
        raise ValueError("OA package is empty.")
    if data.startswith(b"%PDF"):
        return data

    modes: tuple[str, ...] = ("r:gz", "r:")
    last_error: Exception | None = None
    for mode in modes:
        if mode == "r:gz" and not data.startswith(b"\x1f\x8b"):
            continue
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as archive:
                pdf_members = [
                    member
                    for member in archive.getmembers()
                    if member.name.lower().endswith(".pdf")
                ]
                if not pdf_members:
                    raise ValueError("OA package contains no PDF files.")
                extracted = archive.extractfile(pdf_members[0])
                if extracted is None:
                    raise ValueError("Could not extract PDF from OA package.")
                return extracted.read()
        except (tarfile.TarError, zlib.error, OSError, ValueError) as exc:
            last_error = exc

    detail = str(last_error) if last_error else "unrecognized archive format"
    raise ValueError(f"Could not read OA package as tar archive: {detail}")


def resolve_oa_links(pmc_id: str) -> list[tuple[str, str]]:
    """Return ``(format, href)`` pairs from the PMC OA service for one record."""

    response = requests.get(
        OA_URL,
        params={"id": f"PMC{pmc_id}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        return []

    root = ET.fromstring(response.content)
    links: list[tuple[str, str]] = []
    for link_element in root.findall(".//link"):
        link_format = link_element.attrib.get("format", "")
        href = link_element.attrib.get("href")
        if href:
            links.append((link_format, normalize_ftp_path(href)))
    return links


def fetch_pdf_bytes(source_format: str, href: str) -> bytes:
    """Download PDF bytes from an OA ``pdf`` link or by extracting from ``tgz``."""

    if href.startswith("ftp://"):
        payload = ftp_fetch_bytes(href)
        if source_format == "tgz":
            return extract_first_pdf_from_tgz(payload)
        return payload

    response = requests.get(href, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if source_format == "tgz":
        return extract_first_pdf_from_tgz(response.content)
    return response.content


def download_pmc_pdf(pmc_id: str) -> tuple[bytes, str, str]:
    """Resolve and download a PMC PDF; return ``(bytes, source_format, href)``."""

    links = resolve_oa_links(pmc_id)
    if not links:
        raise LookupError("No OA links returned by PMC OA service.")

    preferred_order = ("pdf", "tgz")
    ordered_links = sorted(
        links,
        key=lambda item: preferred_order.index(item[0]) if item[0] in preferred_order else len(preferred_order),
    )

    errors: list[str] = []
    for link_format, href in ordered_links:
        if link_format not in preferred_order:
            continue
        try:
            pdf_bytes = fetch_pdf_bytes(link_format, href)
            if not pdf_bytes.startswith(b"%PDF"):
                raise ValueError("Downloaded payload is not a PDF.")
            return pdf_bytes, link_format, href
        except (OSError, requests.RequestException, ValueError, tarfile.TarError, zlib.error) as exc:
            errors.append(f"{link_format}: {exc}")

    raise RuntimeError("; ".join(errors) if errors else "No downloadable OA PDF or tgz package.")


def relative_pdf_path(pdf_path: Path) -> str:
    """Return ``pdf_path`` relative to ingestion root when possible."""

    try:
        return str(pdf_path.relative_to(INGESTION_ROOT))
    except ValueError:
        return str(pdf_path)


def paper_row(
    *,
    pmc_id: str,
    title: str | None,
    source_format: str | None,
    source_url: str | None,
    pdf_path: Path | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one manifest entry for a PMC paper."""

    row: dict[str, Any] = {
        "pmc_id": f"PMC{pmc_id}",
        "title": title,
        "source_format": source_format,
        "source_url": source_url,
        "pdf_path": relative_pdf_path(pdf_path) if pdf_path else None,
        "status": status,
    }
    if error:
        row["error"] = error
    return row


def run_download(
    *,
    query: str,
    max_results: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Clear output dir, search PMC, download PDFs, return manifest payload."""

    clear_output_dir(output_dir)

    search_retmax = max(max_results, max_results * SEARCH_OVERFETCH_FACTOR)
    print("Querying PMC Open Access database...")
    pmc_ids = search_pmc_ids(query, retmax=search_retmax)
    print(
        f"Found {len(pmc_ids)} candidate papers "
        f"(search cap: {search_retmax}, download target: {max_results})."
    )

    titles = fetch_pmc_titles(pmc_ids)

    papers: list[dict[str, Any]] = []
    downloaded_count = 0
    failed_count = 0
    skipped_count = 0

    for index, pmc_id in enumerate(pmc_ids, start=1):
        if downloaded_count >= max_results:
            break

        title = titles.get(pmc_id)
        title_preview = (title or f"PMC{pmc_id}")[:60]
        filepath = output_dir / f"PMC_{pmc_id}.pdf"
        print(f"[{index}/{len(pmc_ids)}] {title_preview}...")

        try:
            pdf_bytes, source_format, source_url = download_pmc_pdf(pmc_id)
            filepath.write_bytes(pdf_bytes)
            papers.append(
                paper_row(
                    pmc_id=pmc_id,
                    title=title,
                    source_format=source_format,
                    source_url=source_url,
                    pdf_path=filepath,
                    status="downloaded",
                )
            )
            downloaded_count += 1
            print(f"  -> saved {filepath.name} [{downloaded_count}/{max_results}]")

        except LookupError as exc:
            skipped_count += 1
            papers.append(
                paper_row(
                    pmc_id=pmc_id,
                    title=title,
                    source_format=None,
                    source_url=None,
                    pdf_path=None,
                    status="skipped",
                    error=str(exc),
                )
            )
            print(f"  -> skipped: {exc}")

        except (
            RuntimeError,
            requests.RequestException,
            OSError,
            ET.ParseError,
            tarfile.TarError,
            zlib.error,
        ) as exc:
            failed_count += 1
            papers.append(
                paper_row(
                    pmc_id=pmc_id,
                    title=title,
                    source_format=None,
                    source_url=None,
                    pdf_path=filepath,
                    status="failed",
                    error=str(exc),
                )
            )
            print(f"  -> failed: {exc}")

        time.sleep(PMC_REQUEST_DELAY_SECONDS)

    manifest = {
        "source": "pmc",
        "query": query,
        "max_results": max_results,
        "search_retmax": search_retmax,
        "candidate_count": len(pmc_ids),
        "downloaded_count": downloaded_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "paper_count": len(papers),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(
        f"Done: {downloaded_count} downloaded, {failed_count} failed, "
        f"{skipped_count} skipped, {len(pmc_ids)} candidates from search."
    )
    return manifest


def main() -> None:
    """CLI entry for PMC PDF download."""

    parser = argparse.ArgumentParser(
        description="Download PMC open-access PDFs into ingestion/raw_pdfs/",
    )
    parser.add_argument(
        "--query",
        required=True,
        help=(
            "PMC Entrez search query (required). "
            "Include open access[filter] for OA PDFs, e.g. "
            "'(diabetes) AND open access[filter]'."
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=150,
        help="Maximum PDFs to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="raw_pdfs",
        help="Download destination under ingestion/ (default: raw_pdfs).",
    )
    args = parser.parse_args()
    query = validate_pmc_query(args.query)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = INGESTION_ROOT / output_dir

    run_download(
        query=query,
        max_results=args.max_results,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
