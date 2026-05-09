from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
AGENT_TIMEOUT = 600


class RagApiClient:
    """Small HTTP client for the RAG FastAPI service."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def run(self, message: str, session_id: str) -> dict[str, Any]:
        return self._post(
            "/run",
            {"message": message, "session_id": session_id},
        )

    def resume(self, answer: str, session_id: str) -> dict[str, Any]:
        return self._post(
            "/resume",
            {"answer": answer, "session_id": session_id},
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        data.setdefault("retrieved_docs", [])
        return data

