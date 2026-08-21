import logfire
import os
from app.config import settings

logfire.configure(token=settings.LOGFIRE_TOKEN)

# from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
# from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional


# async def startup_event(app: FastAPI)
    # initialize_rails()

    # yield
    # shutdown

app = FastAPI(title= "Enterprise Grade RAG API")

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default user"

@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}

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
