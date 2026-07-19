"""Synchronous HTTP client for the RAG FastAPI service.

Used by tests and scripts; the Dash UI calls the API from browser JavaScript instead.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
AGENT_TIMEOUT = 600


class RagApiClient:
    """Small HTTP client for ``POST /run``, ``/run/stream``, and ``/resume``."""

    def __init__(self, base_url: str | None = None) -> None:
        """Configure base URL and a persistent ``requests.Session``.

        Args:
            base_url: API origin; defaults to ``API_URL`` env or ``DEFAULT_API_URL``.
        """
        self.base_url = (base_url or os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def run(self, message: str, session_id: str) -> dict[str, Any]:
        """Blocking invoke: ``POST /run``.

        Returns:
            Normalized response dict (answer, sources, confidence, cited_document_ids,
            document_catalog, etc.).
        """
        return self.post_json(
            "/run",
            {"message": message, "session_id": session_id},
        )

    def resume(self, answer: str, session_id: str) -> dict[str, Any]:
        """Resume after clarification: ``POST /resume``."""
        return self.post_json(
            "/resume",
            {"answer": answer, "session_id": session_id},
        )

    def iter_run_stream(self, message: str, session_id: str) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON payloads from ``POST /run/stream`` (SSE ``data:`` frames).

        Buffers the byte stream until double-newline frame boundaries, then parses
        each ``data: {...}`` line as JSON.
        """
        url = f"{self.base_url}/run/stream"
        try:
            with self.session.post(
                url,
                json={"message": message, "session_id": session_id},
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=AGENT_TIMEOUT,
            ) as response:
                response.raise_for_status()
                buffer = ""
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
                    if not chunk:
                        continue
                    buffer += chunk.decode("utf-8", errors="replace")
                    while True:
                        sep = buffer.find("\n\n")
                        if sep == -1:
                            break
                        frame = buffer[:sep]
                        buffer = buffer[sep + 2 :]
                        for raw_line in frame.split("\n"):
                            line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
                            if not line.startswith("data: "):
                                continue
                            payload = json.loads(line[6:])
                            yield payload
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError("Cannot reach API. Start uvicorn or set API_URL.") from exc

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to ``path`` and normalize errors into a dict with ``error`` key."""
        try:
            response = self.session.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=AGENT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot reach API. Start uvicorn or set API_URL."}
        except requests.exceptions.Timeout:
            return {"error": f"Request timed out after {AGENT_TIMEOUT}s."}
        except requests.RequestException as exc:
            return {"error": str(exc)}
        except ValueError:
            return {"error": "API response was not valid JSON."}

        if not isinstance(data, dict):
            return {"error": "Unexpected API response shape."}
        data.setdefault("interrupted", False)
        data.setdefault("question", None)
        data.setdefault("answer", None)
        data.setdefault("sources", [])
        data.setdefault("confidence", None)
        data.setdefault("cited_document_ids", [])
        data.setdefault("document_catalog", {})
        return data
