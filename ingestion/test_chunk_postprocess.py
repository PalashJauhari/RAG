"""Tests for ingestion.chunk_postprocess."""

from __future__ import annotations

import copy
import unittest

from ingestion.chunk_postprocess import (
    decode_orig_elements,
    encode_orig_elements,
    extract_images_from_orig_elements,
    normalize_chunk_element,
    strip_image_base64_from_orig_elements,
)


class ChunkPostprocessTests(unittest.TestCase):
    def test_decode_round_trip(self) -> None:
        orig = [
            {"type": "Title", "text": "Section", "element_id": "a1", "metadata": {}},
            {"type": "NarrativeText", "text": "Body", "element_id": "a2", "metadata": {}},
        ]
        encoded = encode_orig_elements(orig)
        self.assertEqual(decode_orig_elements(encoded), orig)

    def test_extract_images_from_orig_elements(self) -> None:
        orig = [
            {
                "type": "Image",
                "element_id": "img-1",
                "metadata": {
                    "page_number": 2,
                    "image_mime_type": "image/jpeg",
                    "image_base64": "abc123",
                },
            },
            {
                "type": "NarrativeText",
                "text": "No image here",
                "metadata": {"page_number": 2},
            },
        ]
        images = extract_images_from_orig_elements(orig)
        self.assertEqual(
            images,
            [
                {
                    "element_id": "img-1",
                    "mime_type": "image/jpeg",
                    "base64": "abc123",
                    "page_number": 2,
                }
            ],
        )

    def test_strip_image_base64_from_orig_elements(self) -> None:
        orig = [
            {
                "type": "Image",
                "element_id": "img-1",
                "metadata": {
                    "image_mime_type": "image/png",
                    "image_base64": "abc123",
                    "page_number": 1,
                },
            }
        ]
        stripped = strip_image_base64_from_orig_elements(orig)
        self.assertNotIn("image_base64", stripped[0]["metadata"])
        self.assertNotIn("image_mime_type", stripped[0]["metadata"])
        self.assertEqual(stripped[0]["metadata"]["page_number"], 1)

    def test_normalize_chunk_element_idempotent(self) -> None:
        orig = [
            {
                "type": "Image",
                "element_id": "img-1",
                "metadata": {
                    "image_mime_type": "image/png",
                    "image_base64": "abc123",
                    "page_number": 3,
                },
            }
        ]
        chunk = {
            "type": "CompositeElement",
            "text": "chunk text",
            "metadata": {"orig_elements": encode_orig_elements(orig)},
        }
        once = normalize_chunk_element(chunk)
        twice = normalize_chunk_element(copy.deepcopy(once))

        self.assertIsInstance(once["metadata"]["orig_elements"], list)
        self.assertEqual(once["metadata"]["images"][0]["base64"], "abc123")
        self.assertNotIn("image_base64", once["metadata"]["orig_elements"][0]["metadata"])
        self.assertEqual(twice, once)


if __name__ == "__main__":
    unittest.main()
