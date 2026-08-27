import logfire
from typing import Optional
from pydantic import BaseModel

from app.config import settings
from fastapi import FastAPI
from app.observability import configure_logfire
from app.agents.graph import rag_agent
# from app.guardrails import initialize_rails, guard

configure_logfire()
app = FastAPI(title="Enterprise Grade RAG API")


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/health")
def check_health():
    return {"message": "Enterprise LangGraph RAG API is running successfully."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid Image of the agents' workflow.
    """
    try:
        pass
    except Exception:
        pass


@app.post("/query")
def query(request: QueryRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    initial_state = {
        "messages": [{"role": "user", "content": request.q}],
        "current_query": "",
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }

    try:
        result = rag_agent.invoke(initial_state, config=config)
    except Exception as e:
        logfire.error(f"Agent invocation failed: {e}")
        return {
            "answer": "Sorry, something went wrong processing your request.",
            "sources": [],
            "thought_process": [],
        }

    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("documents", []),
        "thought_process": result.get("plan", []),
    }