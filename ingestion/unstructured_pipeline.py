"""PDF partition and chunking via Unstructured On-Demand Jobs API.

Run::

    python -m ingestion.unstructured_pipeline
    python -m ingestion.unstructured_pipeline --dry-run

Reads ``manifest.json`` from :envvar:`RAW_PDFS_DIR`, submits PDF batches to
Unstructured, and writes raw chunked elements to :envvar:`CHUNKS_OUTPUT_PATH`.

Phase 2 (embed + Qdrant) is :mod:`ingestion.qdrant_upload`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import CreateJobRequest, DownloadJobOutputRequest
from unstructured_client.models.shared import BodyCreateJob, InputFiles

from ingestion.settings import INGESTION_ROOT, IngestionSettings, settings


def run_pipeline(
    pipeline_settings: IngestionSettings | None = None,
    *,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Partition and chunk downloaded PMC PDFs; write raw Unstructured output."""

    pipeline_settings = pipeline_settings or settings
    raw_pdfs_dir = pipeline_settings.resolve_path(pipeline_settings.raw_pdfs_dir)
    manifest_file = manifest_path or (raw_pdfs_dir / "manifest.json")
    chunks_file = output_path or pipeline_settings.resolve_path(
        pipeline_settings.chunks_output_path
    )

    if not pipeline_settings.unstructured_api_key.strip() and not dry_run:
        raise ValueError("UNSTRUCTURED_API_KEY is required (set in ingestion/.env)")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    papers = [
        row
        for row in manifest.get("papers", [])
        if row.get("status") == "downloaded" and row.get("pdf_path")
    ]

    pdf_paths: list[Path] = []
    for row in papers:
        pdf_path = pipeline_settings.resolve_path(str(row["pdf_path"]))
        if not pdf_path.is_file():
            print(f"Skip missing file: {pdf_path}")
            continue
        size = pdf_path.stat().st_size
        if size > pipeline_settings.unstructured_job_max_file_bytes:
            print(f"Skip oversized ({size} bytes): {pdf_path.name}")
            continue
        pdf_paths.append(pdf_path)

    batch_size = min(10, max(1, pipeline_settings.unstructured_job_batch_size))
    batches: list[list[Path]] = [
        pdf_paths[index : index + batch_size]
        for index in range(0, len(pdf_paths), batch_size)
    ]

    print(f"Manifest: {manifest_file}")
    print(f"Downloaded PDFs eligible: {len(pdf_paths)} in {len(batches)} job batch(es)")

    if dry_run:
        for batch_index, batch in enumerate(batches, start=1):
            names = ", ".join(path.name for path in batch)
            print(f"  batch {batch_index}: {names}")
        return []

    client_kwargs: dict[str, Any] = {"api_key_auth": pipeline_settings.unstructured_api_key}
    if pipeline_settings.unstructured_api_url.strip():
        client_kwargs["server_url"] = pipeline_settings.unstructured_api_url.strip()

    request_data = pipeline_settings.job_request_data()
    results: list[dict[str, Any]] = []
    max_workers = min(
        pipeline_settings.unstructured_job_max_concurrent,
        max(1, len(batches)),
    )

    def process_batch(batch: list[Path]) -> list[dict[str, Any]]:
        input_files: list[InputFiles] = []
        for pdf_path in batch:
            content_type = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
            input_files.append(
                InputFiles(
                    content=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    content_type=content_type,
                )
            )

        with UnstructuredClient(**client_kwargs) as client:
            create_response = client.jobs.create_job(
                request=CreateJobRequest(
                    body_create_job=BodyCreateJob(
                        request_data=request_data,
                        input_files=input_files,
                    )
                )
            )
            job_info = create_response.job_information
            if job_info is None or not job_info.id:
                raise RuntimeError("Unstructured job create returned no job id")

            job_id = job_info.id
            input_file_ids = list(job_info.input_file_ids or [])
            print(f"Job {job_id}: waiting ({len(batch)} file(s))")

            while True:
                status_response = client.jobs.get_job(request={"job_id": job_id})
                job = status_response.job_information
                if job is None:
                    raise RuntimeError(f"Job {job_id}: missing status response")

                status = str(job.status or "").upper()
                if status in {"SCHEDULED", "IN_PROGRESS"}:
                    time.sleep(pipeline_settings.unstructured_job_poll_seconds)
                    continue
                if status != "COMPLETED":
                    raise RuntimeError(f"Job {job_id} ended with status {status!r}")

                batch_results: list[dict[str, Any]] = []
                for file_index, pdf_path in enumerate(batch):
                    file_id = (
                        input_file_ids[file_index]
                        if file_index < len(input_file_ids)
                        else None
                    )
                    if not file_id:
                        print(f"Job {job_id}: no file id for {pdf_path.name}, skipping download")
                        continue

                    download = client.jobs.download_job_output(
                        request=DownloadJobOutputRequest(
                            job_id=job_id,
                            file_id=file_id,
                        )
                    )
                    elements = download.any
                    if isinstance(elements, dict):
                        elements = [elements]
                    batch_results.append(
                        {
                            "filename": pdf_path.name,
                            "elements": elements,
                        }
                    )
                return batch_results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_index, batch in enumerate(batches):
            if batch_index > 0:
                time.sleep(pipeline_settings.unstructured_job_create_interval_seconds)
            futures.append(executor.submit(process_batch, batch))

        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                print(f"Job batch failed: {exc}")

    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} file result(s) to {chunks_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition and chunk PMC PDFs via Unstructured Jobs API.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json (default: ingestion/raw_pdfs/manifest.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for chunks.json (default: ingestion/raw_pdfs/chunks.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List job batches without calling Unstructured.",
    )
    args = parser.parse_args()

    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = INGESTION_ROOT / manifest

    output = args.output
    if output is not None and not output.is_absolute():
        output = INGESTION_ROOT / output

    run_pipeline(manifest_path=manifest, output_path=output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
