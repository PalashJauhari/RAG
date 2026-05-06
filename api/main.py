import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from graph import RetrievalGraph
from output_validation.final_answer import FinalAnswer


class RunRequest(BaseModel):
    session_id: str = Field(description="Stable session id used as LangGraph thread_id.")
    message: str = Field(description="User message to process.")
    recursion_limit: int | None = Field(default=None, description="Optional graph recursion limit.")


class ResumeRequest(BaseModel):
    session_id: str = Field(description="Session id from the interrupted run.")
    answer: str = Field(description="Human answer to the clarification question.")
    recursion_limit: int | None = Field(default=None, description="Optional graph recursion limit.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retrieval_graph = await RetrievalGraph.create()
    try:
        yield
    finally:
        await app.state.retrieval_graph.close()


app = FastAPI(title="RAG Retrieval Orchestrator", lifespan=lifespan)


def get_api_response(session_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build the stable API response from a LangGraph invoke result."""

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
        if isinstance(message, AIMessage) and message.content:
            try:
                payload = json.loads(message.content) if isinstance(message.content, str) else message.content
                answer = FinalAnswer.model_validate(payload)
            except (json.JSONDecodeError, ValueError, TypeError):
                answer = FinalAnswer(answer=str(message.content), sources=[], confidence="low")
            break

    retrieved_docs = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "documents" in payload:
            retrieved_docs = payload["documents"]
            break

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
async def run(request: RunRequest) -> dict[str, Any]:
    result = await app.state.retrieval_graph.run_graph(
        request.session_id,
        request.message,
        request.recursion_limit,
    )
    return get_api_response(request.session_id, result)


@app.post("/resume")
async def resume(request: ResumeRequest) -> dict[str, Any]:
    result = await app.state.retrieval_graph.resume(
        request.session_id,
        request.answer,
        request.recursion_limit,
    )
    return get_api_response(request.session_id, result)

