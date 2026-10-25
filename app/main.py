import uuid
from typing import Literal

import logfire
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

import app.startup_patches
from app.agents.graph import rag_agent
from app.config import settings, validate_env_vars
from app.exceptions import FeedbackError, RAGError
from app.guardrails import input_guard
from app.observability import configure_logfire, instrument_fastapi
from app.services.retrieval.qdrant_service import get_qdrant_client

# from app.guardrails import initialize_rails, guard


app.startup_patches.restore_real_transformers()
configure_logfire()
validate_env_vars()
app = FastAPI(title="Enterprise Grade RAG API")
instrument_fastapi(app)

MAX_QUERY_CHARS = 2000


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    thread_id: str = Field(
        default="default_user",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )

    @field_validator("q")
    @classmethod
    def _q_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("q must not be blank")
        return value


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    thought_process: list[str]
    intent: str
    sources_meta: list[dict] = Field(default_factory=list)
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=128)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


# ---------------------------------------------------------------------------
# Error handling
#
# Every error response has the same shape:
#   {"error": {"code": "<stable_code>", "message": "<safe text>", "error_id": "<id>"}}
# error_id is logged next to the full stack trace, so a user-reported id leads
# straight to the failure in Logfire. Internals are never returned to clients.
# Request validation errors keep FastAPI's standard 422 {"detail": [...]} body.
# ---------------------------------------------------------------------------

def _error_response(status_code: int, code: str, message: str, error_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "error_id": error_id}},
    )


@app.exception_handler(RAGError)
async def handle_rag_error(request: Request, exc: RAGError):
    error_id = uuid.uuid4().hex[:12]
    logfire.error(
        "Request failed: {code}",
        code=exc.code,
        error_id=error_id,
        path=request.url.path,
        detail=exc.detail,
        _exc_info=exc,
    )
    return _error_response(exc.status_code, exc.code, exc.public_message, error_id)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    error_id = uuid.uuid4().hex[:12]
    logfire.error(
        "Unhandled exception",
        error_id=error_id,
        path=request.url.path,
        _exc_info=exc,
    )
    return _error_response(
        500,
        "internal_error",
        "Something went wrong while processing your request.",
        error_id,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def check_health():
    """Liveness: the process is up. Does not touch dependencies."""
    return {"message": "Enterprise LangGraph RAG API is running successfully."}


@app.get("/ready")
def check_ready():
    """Readiness: can we actually serve queries? Verifies Qdrant and the collection."""
    try:
        client = get_qdrant_client(timeout=settings.QDRANT_SEARCH_TIMEOUT_SECONDS)
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            raise RuntimeError(f"Collection '{settings.QDRANT_COLLECTION}' does not exist")
    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        logfire.error("Readiness check failed", error_id=error_id, _exc_info=exc)
        return _error_response(
            503, "not_ready", "The knowledge base is not reachable.", error_id
        )
    return {"status": "ready"}


@app.get("/graph", response_class=PlainTextResponse)
def get_graph():
    """Returns the Mermaid source of the agent workflow (paste into mermaid.live to render)."""
    try:
        return rag_agent.get_graph().draw_mermaid()
    except Exception as exc:
        raise RAGError(f"Could not render agent graph: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    safe_query = input_guard(request.q)
    run_uuid = uuid.uuid4()
    config = {"configurable": {"thread_id": request.thread_id},
              "run_id": run_uuid, "metadata": {"thread_id": request.thread_id}}

    initial_state = {
        "messages": [{"role": "user", "content": safe_query}],
        "current_query": "",
        "intent": "",
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }

    # No blanket try/except here: failures propagate as RAGError (mapped to the right
    # HTTP status by handle_rag_error) or as unexpected errors (500 by the catch-all).
    # The span records the exception either way.
    with logfire.span("rag query", thread_id=request.thread_id, question=safe_query) as span:
        result = rag_agent.invoke(initial_state, config=config)

        answer = result.get("final_answer", "")
        if not answer:
            raise RAGError("Agent finished without producing a final answer")

        intent = result.get("intent", "unknown")
        plan = result.get("plan", [])
        span.set_attribute("intent", intent)
        span.set_attribute("source_count", len(result.get("documents", [])))
        span.set_attribute("cache_hit", "Cache Hit" in plan)

    return QueryResponse(
        answer=answer,
        sources=result.get("documents", []),
        thought_process=plan,
        intent=intent,
        sources_meta=result.get("sources", []),
        trace_id=str(run_uuid),
    )


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        from langsmith import Client
        Client().create_feedback(run_id=request.trace_id, key="user_rating",
                                 score=1 if request.rating == "up" else 0,
                                 comment=request.comment)
        return {"status": "ok"}
    except Exception as exc:
        raise FeedbackError("Unable to persist user feedback") from exc
