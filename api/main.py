import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
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
        if getattr(message, "name", None) != "answer_node":
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


@app.post("/resume")
@observe(name="api_resume")
async def resume(request: ResumeRequest) -> dict[str, Any]:
    result = await app.state.retrieval_graph.resume(
        request.session_id,
        request.answer,
    )
    return get_api_response(request.session_id, result)
