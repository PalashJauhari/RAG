"""FastAPI HTTP surface for the RAG retrieval orchestrator.

Exposes ``POST /run`` (blocking invoke), ``POST /run/stream`` (SSE node progress for
the Dash UI), and ``POST /resume`` (future clarification interrupts). After retrieval,
``recall_check`` routes to answer, intent repair, or partial answer. Application lifespan
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
    """Create the shared ``RetrievalGraph`` once per process and tear it down on shutdown.

    Postgres mode opens ``AsyncPostgresSaver`` via an async context manager stored on
    ``RetrievalGraph`` so ``close()`` can exit the connection cleanly.
    """
    if settings.checkpointer_use_postgres:
        if not settings.database_url.strip():
            raise ValueError("CHECKPOINTER_USE_POSTGRES=true but DATABASE_URL is missing.")
        postgres_context = AsyncPostgresSaver.from_conn_string(settings.database_url)
        checkpointer = await postgres_context.__aenter__()
        await checkpointer.setup()
        app.state.retrieval_graph = RetrievalGraph(checkpointer, postgres_context=postgres_context)
        print("RAG checkpointer: Postgres", flush=True)
    else:
        app.state.retrieval_graph = RetrievalGraph(InMemorySaver())
        print("RAG checkpointer: InMemorySaver", flush=True)
    try:
        yield
    finally:
        await app.state.retrieval_graph.close()


app = FastAPI(title="RAG Retrieval Orchestrator", lifespan=lifespan)

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


def _sse(payload: dict[str, Any]) -> str:
    """Encode one Server-Sent Event ``data:`` frame (JSON payload)."""

    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _message_query_tail(entries: list[Any], max_entries: int = 5) -> list[Any]:
    """Return the last ``max_entries`` ``message_query`` audit rows for compact SSE."""

    if not isinstance(entries, list) or not entries:
        return []
    return entries[-max_entries:]


def get_stream_event(
    session_id: str,
    update: dict[str, Any],
    *,
    retrieved_doc_count: int | None = None,
    new_doc_count: int | None = None,
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
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "fact_decomposition":
        required_facts = payload.get("required_facts") or []
        event.update(
            {
                "label": "Decomposing required facts",
                "required_facts": required_facts,
                "required_fact_count": len(required_facts),
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "query_complexity":
        query_complexity = payload.get("query_complexity") or {}
        event.update(
            {
                "label": "Classifying query complexity",
                "complexity": query_complexity.get("complexity"),
                "explanation": query_complexity.get("explanation"),
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "query_splitter":
        event.update(
            {
                "label": "Preparing retrieval queries",
                "active_retrieval_queries": payload.get("active_retrieval_queries") or [],
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "retrieval":
        event.update(
            {
                "label": "Retrieving documents",
                "retrieval_strategy": payload.get("retrieval_strategy"),
                "retrieval_queries": payload.get("active_retrieval_queries") or [],
                "new_doc_count": new_doc_count or 0,
                "retrieved_doc_count": retrieved_doc_count or 0,
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "recall_check":
        verified_facts = payload.get("verified_facts") or []
        unsupported_facts = [
            row for row in verified_facts if isinstance(row, dict) and not row.get("verification_status")
        ]
        event.update(
            {
                "label": "Checking recall",
                "recall_sufficient": payload.get("recall_sufficient"),
                "required_facts": payload.get("required_facts") or [],
                "verified_facts": verified_facts,
                "unsupported_facts": unsupported_facts,
                "unsupported_fact_count": len(unsupported_facts),
                "retrieved_doc_count": retrieved_doc_count or 0,
                "retrieval_retry_count": payload.get("retrieval_retry_count", 0),
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "intent_check":
        event.update(
            {
                "label": "Checking intent alignment",
                "intent_aligned": payload.get("intent_aligned"),
                "fact_intents": payload.get("fact_intents") or [],
                "intent_mismatch_details": payload.get("intent_mismatch_details") or [],
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "gap_fill":
        event.update(
            {
                "label": "Filling recall gaps",
                "active_retrieval_queries": payload.get("active_retrieval_queries") or [],
            }
        )
    elif node_name == "strategy_upgrade":
        event.update(
            {
                "label": "Evaluating retrieval tier",
                "retrieval_strategy": payload.get("retrieval_strategy"),
                "retrieval_retry_count": payload.get("retrieval_retry_count", 0),
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "intent_correction_rewriter":
        event.update(
            {
                "label": "Correcting retrieval intent",
                "active_retrieval_queries": payload.get("active_retrieval_queries") or [],
            }
        )
        mq = payload.get("message_query")
        if mq:
            event["message_query_tail"] = _message_query_tail(mq)
    elif node_name == "clear_turn_trace":
        event.update(
            {
                "label": "Clearing turn trace",
            }
        )
    elif node_name in {"answer", "partial_answer"}:
        # Final answer lives in messages as JSON; parse for the ``final`` SSE frame.
        answer = FinalAnswer(answer="", sources=[], confidence="low")
        for message in payload.get("messages") or []:
            if not isinstance(message, AIMessage) or not message.content:
                continue
            try:
                content = (
                    json.loads(message.content)
                    if isinstance(message.content, str)
                    else message.content
                )
                answer = FinalAnswer.model_validate(content)
                break
            except (json.JSONDecodeError, ValueError, TypeError):
                answer = FinalAnswer(answer=str(message.content), sources=[], confidence="low")
                break
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

    messages = result.get("messages", [])
    answer = FinalAnswer(answer="", sources=[], confidence="low")
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or not message.content:
            continue
        if getattr(message, "name", None) not in {"answer_node", "partial_answer_node"}:
            continue
        try:
            payload = json.loads(message.content) if isinstance(message.content, str) else message.content
            answer = FinalAnswer.model_validate(payload)
            break
        except (json.JSONDecodeError, ValueError, TypeError):
            answer = FinalAnswer(answer=str(message.content), sources=[], confidence="low")
            break

    if not answer.answer:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                answer = FinalAnswer(answer=str(message.content), sources=[], confidence="low")
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
        try:
            async for update in app.state.retrieval_graph.stream_run(
                request.session_id,
                request.message,
            ):
                new_doc_count = 0
                if isinstance(update, dict) and update:
                    node_name = next(iter(update))
                    payload = update.get(node_name) or {}
                    if node_name == "retrieval":
                        docs = payload.get("retrieved_documents") or []
                        new_doc_count = len(docs) if isinstance(docs, list) else 0
                        retrieved_doc_count += new_doc_count
                yield _sse(
                    get_stream_event(
                        request.session_id,
                        update,
                        retrieved_doc_count=retrieved_doc_count,
                        new_doc_count=new_doc_count,
                    )
                )
            yield _sse(
                {
                    "type": "done",
                    "session_id": request.session_id,
                    "retrieved_doc_count": retrieved_doc_count,
                }
            )
        except Exception as exc:
            # Streaming responses cannot switch to a normal JSON error once started.
            yield _sse(
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
