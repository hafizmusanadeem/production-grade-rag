import logfire
from typing import Optional
from pydantic import BaseModel

from app.config import settings
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
# from app.guardrails import initialize_rails, guard


logfire.configure(token=settings.LOGFIRE_TOKEN)
app = FastAPI(title= "Enterprise Grade RAG API")

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default user"

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
    except Exception as e: 
        pass

@app.post("/query/{query})
def query()
