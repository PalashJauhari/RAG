"""FastAPI HTTP surface for the RAG retrieval orchestrator.

Exposes ``POST /run`` (blocking invoke), ``POST /run/stream`` (SSE node progress for
the Dash UI), and ``POST /resume`` (future clarification interrupts). After retrieval, ``recall_check`` routes to answer, gap-fill repair, or partial answer. Application lifespan
constructs a :class:`~graph.graph.RetrievalGraph` with either in-memory or Postgres
LangGraph checkpointing. ``session_id`` maps to LangGraph ``thread_id``.
"""

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from config.settings import settings
from graph import RetrievalGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from output_validation.final_answer import FinalAnswer


# --- Request models ---


class RunRequest(BaseModel):
    """Body for ``POST /run`` and ``POST /run/stream``."""

    session_id: str = Field(description="Stable session id used as conversation thread id for checkpointing.")
    message: str = Field(description="User message to process.")


class ResumeRequest(BaseModel):
    """Body for ``POST /resume`` when the graph has interrupted for clarification."""

    session_id: str = Field(description="Session id from the interrupted run.")
    answer: str = Field(description="Human answer to the clarification question.")


# --- Application lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the shared ``RetrievalGraph`` once per process and tear it down on shutdown."""
    postgres_context = None
    if settings.checkpointer_use_postgres:
        if not settings.database_url.strip():
            raise ValueError("CHECKPOINTER_USE_POSTGRES=true but DATABASE_URL is missing.")
        postgres_context = AsyncPostgresSaver.from_conn_string(settings.database_url)
        checkpointer = await postgres_context.__aenter__()
        await checkpointer.setup()
        app.state.retrieval_graph = RetrievalGraph(checkpointer)
        print("RAG checkpointer: Postgres", flush=True)
    else:
        app.state.retrieval_graph = RetrievalGraph(InMemorySaver())
        print("RAG checkpointer: InMemorySaver", flush=True)
    try:
        yield
    finally:
        await app.state.retrieval_graph.retriever.qdrant.close()
        if postgres_context is not None:
            await postgres_context.__aexit__(None, None, None)


app = FastAPI(title="Factline", lifespan=lifespan)

# CORS: Dash on 8050 calls the API from the browser; allow loopback variants.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8050",
        "http://localhost:8050",
        # Browsers often send these when Dash binds to 0.0.0.0 or IPv6 loopback.
        "http://0.0.0.0:8050",
        "http://[::1]:8050",
    ],
    # Any localhost-style origin with an explicit port (e.g. different Dash port).
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost|\[::1\]|0\.0\.0\.0)(:[1-9]\d*)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SSE helpers ---


def encode_sse_frame(payload: dict[str, Any]) -> str:
    """Encode one Server-Sent Event ``data:`` frame (JSON payload)."""

    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def message_content(message: Any) -> str | None:
    """Extract string content from an ``AIMessage`` or a LangGraph/LangChain message dict."""

    raw: Any = None
    if isinstance(message, AIMessage):
        raw = message.content
    elif isinstance(message, dict):
        raw = message.get("content")
        if raw is None:
            data = message.get("data")
            if isinstance(data, dict):
                raw = data.get("content")
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        joined = "\n".join(parts).strip()
        return joined or None
    return str(raw).strip() or None


def message_name(message: Any) -> str | None:
    """Return graph node name from ``AIMessage.name`` or serialized message dict."""

    if isinstance(message, AIMessage):
        name = getattr(message, "name", None)
        return str(name).strip() if name else None
    if isinstance(message, dict):
        name = message.get("name")
        if name:
            return str(name).strip()
        data = message.get("data")
        if isinstance(data, dict) and data.get("name"):
            return str(data["name"]).strip()
    return None


def final_answer_from_messages(
    messages: list[Any],
    *,
    allowed_names: set[str] | None = None,
) -> FinalAnswer:
    """Parse ``FinalAnswer`` JSON from the first usable AI message in ``messages``."""

    fallback = FinalAnswer(answer="", sources=[], confidence="low")
    for message in messages or []:
        if allowed_names is not None:
            name = message_name(message)
            if name and name not in allowed_names:
                continue
        content = message_content(message)
        if not content:
            continue
        try:
            payload = json.loads(content) if content.lstrip().startswith("{") else content
            if isinstance(payload, dict):
                return FinalAnswer.model_validate(payload)
            return FinalAnswer(answer=str(payload), sources=[], confidence="low")
        except (json.JSONDecodeError, ValueError, TypeError):
            return FinalAnswer(answer=content, sources=[], confidence="low")
    return fallback


def get_stream_event(
    session_id: str,
    update: dict[str, Any],
    *,
    retrieved_doc_count: int | None = None,
    new_doc_count: int | None = None,
    retrieval_loop_count: int = 0,
) -> dict[str, Any]:
    """Map a LangGraph ``stream_mode='updates'`` chunk to a stable SSE event dict.

    The Dash clientside script keys off ``node``, ``type``, and node-specific fields
    documented in the README. Keep field names backward-compatible when changing this.

    Args:
        session_id: Conversation thread id echoed on every frame.
        update: Single-node partial state update from ``astream``.

    Returns:
        JSON-serializable event with ``type`` of ``node``, ``final``, or ``debug``.
    """

    if not isinstance(update, dict) or not update:
        return {
            "type": "debug",
            "session_id": session_id,
            "payload": update,
        }

    node_name = next(iter(update))
    payload = update.get(node_name) or {}
    event: dict[str, Any] = {
        "type": "node",
        "session_id": session_id,
        "node": node_name,
        "status": "completed",
    }

    # --- Per-node SSE field mapping (mirrors graph node outputs) ---

    if node_name == "query_normalisation":
        event.update(
            {
                "label": "Normalizing query",
                "normalized_query": payload.get("normalized_query"),
            }
        )
    elif node_name == "fact_decomposition":
        facts = payload.get("facts") or []
        event.update(
            {
                "label": "Decomposing facts",
                "fact_count": len(facts),
            }
        )
    elif node_name == "query_complexity":
        event.update(
            {
                "label": "Routing by fact count",
                "needs_split": payload.get("needs_split"),
            }
        )
    elif node_name == "query_splitter":
        event.update(
            {
                "label": "Preparing retrieval queries",
                "active_retrieval_queries": payload.get("active_retrieval_queries") or [],
            }
        )
    elif node_name == "retrieval":
        event.update(
            {
                "label": "Retrieving documents",
                "retrieval_strategy": payload.get("retrieval_strategy"),
                "retrieval_queries": payload.get("active_retrieval_queries") or [],
                "new_doc_count": new_doc_count or 0,
                "retrieved_doc_count": retrieved_doc_count or 0,
                "retrieval_loop_count": retrieval_loop_count,
            }
        )
    elif node_name == "recall_check":
        facts = payload.get("facts") or []
        unsupported = [
            row for row in facts if isinstance(row, dict) and not row.get("verification_status")
        ]
        event.update(
            {
                "label": "Checking recall",
                "recall_sufficient": payload.get("recall_sufficient"),
                "unsupported_fact_count": len(unsupported),
                "retrieved_doc_count": retrieved_doc_count or 0,
                "retrieval_loop_count": retrieval_loop_count,
            }
        )
    elif node_name == "create_queries_for_unsupported_facts":
        event.update(
            {
                "label": "Creating queries for unsupported facts",
                "active_retrieval_queries": payload.get("active_retrieval_queries") or [],
            }
        )
    elif node_name == "strategy_upgrade":
        event.update(
            {
                "label": "Evaluating retrieval tier",
                "retrieval_strategy": payload.get("retrieval_strategy"),
                "retrieval_loop_count": retrieval_loop_count,
            }
        )
    elif node_name in {"answer", "partial_answer"}:
        # Final answer lives in messages as JSON; parse for the ``final`` SSE frame.
        allowed = {"answer_node"} if node_name == "answer" else {"partial_answer_node"}
        answer = final_answer_from_messages(
            list(payload.get("messages") or []),
            allowed_names=allowed,
        )
        event.update(
            {
                "type": "final",
                "label": "Answer ready" if node_name == "answer" else "Partial answer ready",
                "answer": answer.answer,
                "sources": answer.sources,
                "confidence": answer.confidence,
                "retrieved_doc_count": retrieved_doc_count or 0,
            }
        )
    else:
        event.update(
            {
                "label": node_name.replace("_", " ").title(),
            }
        )

    return event


def get_api_response(session_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a completed graph invoke into the stable ``/run`` JSON shape.

    Prefer ``AIMessage`` payloads from ``answer_node`` or ``partial_answer_node`` (by
    ``message.name``). Fall back to the last AI message if parsing fails. Interrupt
    payloads surface ``interrupted`` and ``question`` for future ``/resume``.

    Args:
        session_id: Thread id echoed in the response.
        result: Final state dict from ``RetrievalGraph.run`` or ``resume``.

    Returns:
        Dict with ``answer``, ``sources``, ``confidence``, and ``retrieved_docs``.
    """

    interrupts = result.get("__interrupt__") or []
    if interrupts:
        value = getattr(interrupts[0], "value", interrupts[0])
        question = value.get("question") if isinstance(value, dict) else str(value)
        return {
            "session_id": session_id,
            "interrupted": True,
            "question": question,
            "answer": None,
            "sources": [],
            "confidence": None,
            "retrieved_docs": [],
        }

    messages = list(result.get("messages") or [])
    answer = final_answer_from_messages(
        list(reversed(messages)),
        allowed_names={"answer_node", "partial_answer_node"},
    )
    if not answer.answer:
        for message in reversed(messages):
            content = message_content(message)
            if content:
                answer = FinalAnswer(answer=content, sources=[], confidence="low")
                break

    retrieved_docs = list(result.get("retrieved_documents") or [])

    return {
        "session_id": session_id,
        "interrupted": False,
        "question": None,
        "answer": answer.answer,
        "sources": answer.sources,
        "confidence": answer.confidence,
        "retrieved_docs": retrieved_docs,
    }


# --- HTTP routes (session_id == LangGraph thread_id) ---


@app.post("/run")
async def run(request: RunRequest) -> dict[str, Any]:
    """Run one user turn to completion and return the final answer payload."""
    result = await app.state.retrieval_graph.run(
        request.session_id,
        request.message,
    )
    return get_api_response(request.session_id, result)


@app.post("/run/stream")
async def run_stream(request: RunRequest) -> StreamingResponse:
    """Stream node-level graph progress as Server-Sent Events (``text/event-stream``)."""

    async def event_generator():
        # Map each LangGraph stream chunk to SSE; expose counts only to keep UI payloads light.
        retrieved_doc_count = 0
        retrieval_loop_count = 0
        try:
            async for update in app.state.retrieval_graph.stream_run(
                request.session_id,
                request.message,
            ):
                new_doc_count = 0
                if isinstance(update, dict) and update:
                    node_name = next(iter(update))
                    payload = update.get(node_name) or {}
                    if node_name == "strategy_upgrade":
                        retrieval_loop_count = int(
                            payload.get("retrieval_retry_count", retrieval_loop_count)
                        )
                    if node_name == "retrieval":
                        docs = payload.get("retrieved_documents") or []
                        new_doc_count = len(docs) if isinstance(docs, list) else 0
                        retrieved_doc_count += new_doc_count
                yield encode_sse_frame(
                    get_stream_event(
                        request.session_id,
                        update,
                        retrieved_doc_count=retrieved_doc_count,
                        new_doc_count=new_doc_count,
                        retrieval_loop_count=retrieval_loop_count,
                    )
                )
            yield encode_sse_frame(
                {
                    "type": "done",
                    "session_id": request.session_id,
                    "retrieved_doc_count": retrieved_doc_count,
                }
            )
        except Exception as exc:
            # Streaming responses cannot switch to a normal JSON error once started.
            yield encode_sse_frame(
                {
                    "type": "error",
                    "session_id": request.session_id,
                    "error": str(exc),
                }
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/resume")
async def resume(request: ResumeRequest) -> dict[str, Any]:
    """Resume a paused graph after a human clarification (when interrupts are enabled)."""
    result = await app.state.retrieval_graph.resume(
        request.session_id,
        request.answer,
    )
    return get_api_response(request.session_id, result)
