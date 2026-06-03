"""Ingestion settings loaded from ``ingestion/.env`` only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

INGESTION_ROOT = Path(__file__).resolve().parent


class IngestionConfig(BaseSettings):
    """PMC ingestion: Unstructured, Qdrant, OpenAI, Jina (standalone from app/benchmark)."""

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    ingestion_enrichment_model: str = Field(
        default="gpt-4o-mini",
        alias="INGESTION_ENRICHMENT_MODEL",
    )

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = ""
    qdrant_dense_vector_name: str = "dense"
    qdrant_bm25_vector_name: str = "bm25"
    qdrant_colbert_vector_name: str = "colbert"
    qdrant_bm25_model: str = "Qdrant/bm25"

    use_bm25: bool = True
    use_late_interaction: bool = True

    jina_api_key: str = ""
    jina_colbert_model: str = "jina-colbert-v2"
    jina_colbert_dimensions: int = 128
    jina_multi_vector_url: str = "https://api.jina.ai/v1/multi-vector"

    request_timeout_seconds: int = 60
    ingestion_upload_batch_size: int = 16

    unstructured_api_key: str = ""
    unstructured_api_url: str = ""

    unstructured_job_batch_size: int = 10
    unstructured_job_poll_seconds: int = 10
    unstructured_job_max_concurrent: int = 5
    unstructured_job_create_interval_seconds: int = 1
    unstructured_job_max_file_bytes: int = 10_485_760

    raw_pdfs_dir: str = "raw_pdfs"
    chunks_output_path: str = "raw_pdfs/chunks.json"

    unstructured_partition_strategy: str = "hi_res"
    unstructured_infer_table_structure: bool = True
    unstructured_include_page_breaks: bool = False
    unstructured_coordinates: bool = False
    unstructured_ocr_languages: str = "eng"
    unstructured_exclude_elements: str = ""
    unstructured_extract_image_block_types: str = "Image"
    unstructured_encoding: str = "utf-8"

    unstructured_image_description_enabled: bool = True
    unstructured_image_description_subtype: str = "openai_image_description"
    unstructured_image_description_provider: str = "openai"
    unstructured_image_description_model: str = "gpt-4o-mini"

    unstructured_chunk_strategy: str = "chunk_by_title"
    unstructured_max_characters: int = 2000
    unstructured_new_after_n_chars: int = 1500
    unstructured_overlap: int = 200
    unstructured_overlap_all: bool = False
    unstructured_combine_text_under_n_chars: int = 400
    unstructured_multipage_sections: bool = True
    unstructured_include_orig_elements: bool = True
    unstructured_contextual_chunking: str = ""

    model_config = SettingsConfigDict(
        env_file=INGESTION_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to ``ingestion/`` unless already absolute."""

        path = Path(relative)
        return path if path.is_absolute() else INGESTION_ROOT / path

    @property
    def manifest_path(self) -> Path:
        return self.resolve_path(self.raw_pdfs_dir) / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.resolve_path(self.chunks_output_path)

    def build_job_nodes(self) -> list[dict[str, Any]]:
        """Build Unstructured on-demand job DAG from env settings."""

        exclude = [
            item.strip()
            for item in self.unstructured_exclude_elements.split(",")
            if item.strip()
        ]
        ocr_languages = [
            item.strip()
            for item in self.unstructured_ocr_languages.split(",")
            if item.strip()
        ]
        extract_image_block_types = [
            item.strip()
            for item in self.unstructured_extract_image_block_types.split(",")
            if item.strip()
        ]

        partitioner_settings: dict[str, Any] = {
            "strategy": self.unstructured_partition_strategy,
            "infer_table_structure": self.unstructured_infer_table_structure,
            "include_page_breaks": self.unstructured_include_page_breaks,
            "coordinates": self.unstructured_coordinates,
            "ocr_languages": ocr_languages or ["eng"],
            "encoding": self.unstructured_encoding,
        }
        if exclude:
            partitioner_settings["exclude_elements"] = exclude
        if extract_image_block_types:
            partitioner_settings["extract_image_block_types"] = extract_image_block_types

        nodes: list[dict[str, Any]] = [
            {
                "name": "Partitioner",
                "type": "partition",
                "subtype": "unstructured_api",
                "settings": partitioner_settings,
            }
        ]

        if self.unstructured_image_description_enabled:
            nodes.append(
                {
                    "name": "ImageDescription",
                    "type": "prompter",
                    "subtype": self.unstructured_image_description_subtype,
                    "settings": {
                        "provider_type": self.unstructured_image_description_provider,
                        "model": self.unstructured_image_description_model,
                    },
                }
            )

        chunker_settings: dict[str, Any] = {
            "multipage_sections": self.unstructured_multipage_sections,
            "combine_text_under_n_chars": self.unstructured_combine_text_under_n_chars,
            "include_orig_elements": self.unstructured_include_orig_elements,
            "new_after_n_chars": self.unstructured_new_after_n_chars,
            "max_characters": self.unstructured_max_characters,
            "overlap": self.unstructured_overlap,
            "overlap_all": self.unstructured_overlap_all,
        }
        contextual = self.unstructured_contextual_chunking.strip().lower()
        if contextual in {"v1", "true", "1", "yes", "on"}:
            chunker_settings["contextual_chunking_strategy"] = "v1"

        nodes.append(
            {
                "name": "Chunker",
                "type": "chunk",
                "subtype": self.unstructured_chunk_strategy,
                "settings": chunker_settings,
            }
        )
        return nodes

    def job_request_data(self) -> str:
        """JSON string for ``request_data`` field on job create."""

        return json.dumps({"job_nodes": self.build_job_nodes()})


def load_ingestion_config() -> IngestionConfig:
    """Load settings from ``ingestion/.env``."""

    return IngestionConfig()
