import json
import os
from typing import Any

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from config.settings import settings
from graph.graph import graph


class RunRequest(BaseModel):
    session_id: str = Field(description="Stable session id used as LangGraph thread_id.")
    message: str = Field(description="User message to process.")
    recursion_limit: int = Field(default=25, description="LangGraph recursion limit.")


class ResumeRequest(BaseModel):
    session_id: str = Field(description="Session id from the interrupted run.")
    answer: str = Field(description="Human answer to the clarification question.")
    recursion_limit: int = Field(default=25, description="LangGraph recursion limit.")


app = FastAPI(title="RAG Retrieval Orchestrator")

langfuse_handler = None
if settings.langfuse_tracing_enabled:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "true")

    from langfuse.langchain import CallbackHandler

    langfuse_handler = CallbackHandler()


def get_api_response(result: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Return a stable API shape for normal graph output and HITL interrupts."""

    interrupts = result.get("__interrupt__")
    if interrupts:
        interrupt_value = getattr(interrupts[0], "value", interrupts[0])
        question = (
            interrupt_value.get("question")
            if isinstance(interrupt_value, dict)
            else str(interrupt_value)
        )
        return {
            "status": "interrupted",
            "session_id": session_id,
            "question": question,
            "interrupt": interrupt_value,
        }

    messages = result.get("messages", [])
    content = getattr(messages[-1], "content", "") if messages else ""
    try:
        response = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        response = {"answer": content, "sources": [], "confidence": "low"}

    return {
        "status": "completed",
        "session_id": session_id,
        "response": response,
    }


@app.post("/run")
async def run(request: RunRequest) -> dict[str, Any]:
    graph_config: dict[str, Any] = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": request.recursion_limit,
        "metadata": {"session_id": request.session_id},
    }
    if langfuse_handler:
        graph_config["callbacks"] = [langfuse_handler]

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=graph_config,
    )
    return get_api_response(result, request.session_id)


@app.post("/resume")
async def resume(request: ResumeRequest) -> dict[str, Any]:
    graph_config: dict[str, Any] = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": request.recursion_limit,
        "metadata": {"session_id": request.session_id},
    }
    if langfuse_handler:
        graph_config["callbacks"] = [langfuse_handler]

    result = await graph.ainvoke(
        Command(resume=request.answer),
        config=graph_config,
    )
    return get_api_response(result, request.session_id)
