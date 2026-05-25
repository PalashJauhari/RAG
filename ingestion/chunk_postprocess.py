"""Post-process Unstructured chunk output: decode orig_elements and extract images."""

from __future__ import annotations

import base64
import copy
import json
import zlib
from typing import Any


def decode_orig_elements(encoded: str) -> list[dict[str, Any]]:
    """Decode Unstructured ``metadata.orig_elements`` (base64 zlib JSON)."""

    if not str(encoded or "").strip():
        return []
    decoded = base64.b64decode(encoded)
    decompressed = zlib.decompress(decoded)
    payload = json.loads(decompressed.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("orig_elements must decode to a JSON array")
    return payload


def encode_orig_elements(orig_elements: list[dict[str, Any]]) -> str:
    """Encode element list the way Unstructured serializes orig_elements (for tests)."""

    json_bytes = json.dumps(orig_elements, sort_keys=True).encode("utf-8")
    return base64.b64encode(zlib.compress(json_bytes)).decode("utf-8")


def extract_images_from_orig_elements(
    orig_elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pull base64 Image blocks from decoded orig_elements."""

    images: list[dict[str, Any]] = []
    for element in orig_elements:
        if element.get("type") != "Image":
            continue
        metadata = element.get("metadata") or {}
        base64_data = metadata.get("image_base64")
        if not base64_data:
            continue
        image: dict[str, Any] = {
            "element_id": element.get("element_id"),
            "mime_type": metadata.get("image_mime_type") or "image/png",
            "base64": base64_data,
        }
        page_number = metadata.get("page_number")
        if page_number is not None:
            image["page_number"] = page_number
        images.append(image)
    return images


def strip_image_base64_from_orig_elements(
    orig_elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove bulky image bytes from Image metadata after extraction."""

    stripped: list[dict[str, Any]] = []
    for element in orig_elements:
        item = copy.deepcopy(element)
        if item.get("type") == "Image":
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop("image_base64", None)
                metadata.pop("image_mime_type", None)
        stripped.append(item)
    return stripped


def normalize_chunk_element(element: dict[str, Any]) -> dict[str, Any]:
    """Decode orig_elements, extract images, and strip duplicated base64."""

    normalized = copy.deepcopy(element)
    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        return normalized

    orig_elements = metadata.get("orig_elements")
    if not isinstance(orig_elements, str):
        return normalized

    decoded = decode_orig_elements(orig_elements)
    images = extract_images_from_orig_elements(decoded)
    metadata["orig_elements"] = strip_image_base64_from_orig_elements(decoded)
    if images:
        metadata["images"] = images
    else:
        metadata.pop("images", None)
    return normalized


def normalize_file_result(file_result: dict[str, Any]) -> dict[str, Any]:
    """Normalize every chunk element in one PDF file result."""

    normalized = copy.deepcopy(file_result)
    elements = normalized.get("elements")
    if not isinstance(elements, list):
        return normalized
    normalized["elements"] = [
        normalize_chunk_element(element) if isinstance(element, dict) else element
        for element in elements
    ]
    return normalized
