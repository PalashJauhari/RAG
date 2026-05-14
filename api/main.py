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
from observability.langfuse_handler import get_observe
from output_validation.final_answer import FinalAnswer

observe = get_observe()


class RunRequest(BaseModel):
    session_id: str = Field(description="Stable session id used as conversation thread id for checkpointing.")
    message: str = Field(description="User message to process.")


class ResumeRequest(BaseModel):
    session_id: str = Field(description="Session id from the interrupted run.")
    answer: str = Field(description="Human answer to the clarification question.")


@asynccontextmanager
async def lifespan(app: FastAPI):
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8050",
        "http://localhost:8050",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(payload: dict[str, Any]) -> str:
    """Encode one Server-Sent Event data frame."""

    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def get_stream_event(session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    """Convert raw LangGraph node updates into stable, compact stream events."""

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

    if node_name == "query_normalisation":
        event.update(
            {
                "label": "Normalizing query",
                "normalized_query": payload.get("normalized_query"),
            }
        )
    elif node_name == "query_complexity":
        query_complexity = payload.get("query_complexity") or {}
        event.update(
            {
                "label": "Classifying query complexity",
                "complexity": query_complexity.get("complexity"),
                "explanation": query_complexity.get("explanation"),
            }
        )
    elif node_name in {"query_splitter", "query_expansion", "query_rewriter"}:
        event.update(
            {
                "label": "Preparing retrieval queries",
                "parsed_queries": payload.get("parsed_queries") or [],
            }
        )
    elif node_name == "retrieval":
        retrieved_documents = payload.get("retrieved_documents") or []
        event.update(
            {
                "label": "Retrieving documents",
                "retrieval_query_source": payload.get("retrieval_query_source"),
                "retrieval_queries": payload.get("last_retrieval_queries") or [],
                "retrieved_doc_count": len(retrieved_documents),
            }
        )
    elif node_name == "information_evaluator":
        evaluation = payload.get("information_evaluation") or {}
        event.update(
            {
                "label": "Evaluating evidence",
                "evaluation_status": evaluation.get("evaluation_status"),
                "missing_evidence_details": payload.get("missing_evidence_details") or [],
                "insufficient_recall_retry_count": payload.get(
                    "insufficient_recall_retry_count",
                    0,
                ),
                "intent_mismatch_retry_count": payload.get(
                    "intent_mismatch_retry_count",
                    0,
                ),
            }
        )
    elif node_name == "gap_fill":
        event.update(
            {
                "label": "Filling recall gaps",
                "parsed_queries_insufficient_recall": payload.get(
                    "parsed_queries_insufficient_recall"
                )
                or [],
            }
        )
    elif node_name == "intent_correction_rewriter":
        event.update(
            {
                "label": "Correcting retrieval intent",
                "parsed_queries_intent_correction": payload.get(
                    "parsed_queries_intent_correction"
                )
                or [],
            }
        )
    elif node_name in {"answer", "partial_answer"}:
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
    """Build the stable API response from a completed graph invoke result."""

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


@app.post("/run")
@observe(name="api_run")
async def run(request: RunRequest) -> dict[str, Any]:
    result = await app.state.retrieval_graph.run(
        request.session_id,
        request.message,
    )
    return get_api_response(request.session_id, result)


@app.post("/run/stream")
@observe(name="api_run_stream")
async def run_stream(request: RunRequest) -> StreamingResponse:
    """Stream node-level graph progress without changing the stable /run endpoint."""

    async def event_generator():
        last_retrieved_docs: list[Any] = []
        try:
            async for update in app.state.retrieval_graph.stream_run(
                request.session_id,
                request.message,
            ):
                if isinstance(update, dict) and update:
                    node_name = next(iter(update))
                    payload = update.get(node_name) or {}
                    if node_name == "retrieval":
                        docs = payload.get("retrieved_documents")
                        if docs is not None:
                            last_retrieved_docs = list(docs)
                yield _sse(get_stream_event(request.session_id, update))
            yield _sse(
                {
                    "type": "done",
                    "session_id": request.session_id,
                    "retrieved_docs": last_retrieved_docs,
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
@observe(name="api_resume")
async def resume(request: ResumeRequest) -> dict[str, Any]:
    result = await app.state.retrieval_graph.resume(
        request.session_id,
        request.answer,
    )
    return get_api_response(request.session_id, result)
