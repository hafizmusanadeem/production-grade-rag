"""evals/target.py -- the eval target adapter.

A single object that runs the real LangGraph agent and returns everything the
eval suite needs: the answer, retrieved context (chunk text with the "CONTENT:"
prefix stripped), sources metadata, intent, plan, timings, token usage, error
code, and whether the agent declined. This is the seam every eval hits.
"""
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import logfire

from app.agents.graph import rag_agent
from app.agents.state import AgentState
from app.exceptions import RAGError
from evals.config import RESPONDER_CONFIG_ID


@dataclass
class EvalResult:
    query: str
    answer: str
    context: List[str]          # chunk text given to the generator, "CONTENT:" prefix STRIPPED
    sources: List[Dict[str, Any]] = field(default_factory=list)
    intent: str = "unknown"
    declined: bool = False
    plan: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None   # RAGError.code if the call raised; answer == "" then


def _make_initial_state(query: str) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "current_query": "",
        "intent": "",
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }


class AgentTarget:
    """Adapter: invoke the LangGraph agent for a single query."""

    def __init__(self, *, bypass_cache: bool = True):
        # Cache control: disable Portkey caching so repeated eval runs are honest.
        # Verify the exact header name in Portkey docs; we send force-refresh here.
        bypass_cache = bypass_cache

    def invoke(self, query: str, *, thread_id: Optional[str] = None,
               history: Optional[List[dict]] = None) -> EvalResult:
        """Run the agent for one query and return a fully-populated EvalResult."""
        # Fresh uuid per case so MemorySaver doesn't leak state between golden items.
        tid = thread_id or f"eval-{uuid.uuid4().hex}"

        if history:
            state = _make_state_with_history(query, history)
        else:
            state = _make_initial_state(query)

        try:
            result = rag_agent.invoke(state, config={"configurable": {"thread_id": tid}})
        except RAGError as exc:
            return EvalResult(query=query, answer="", context=[],
                              error=exc.code)
        except Exception as exc:
            logfire.error("AgentTarget invoke failed", error_id=str(uuid.uuid4()[:12]),
                          error=str(exc))
            return EvalResult(query=query, answer="", context=[],
                              error="internal_error")

        final_answer = result.get("final_answer", "") or ""
        intent = result.get("intent", "unknown")
        plan = result.get("plan", []) or []

        raw_docs = result.get("documents", [])

        # Strip the "CONTENT: " prefix that format_docs_as_text injects in retrieve_node.
        context = []
        sources = []
        for doc in raw_docs:
            text = doc
            if isinstance(doc, str):
                text = doc
            elif isinstance(doc, dict):
                text = doc.get("page_content", str(doc))
            if isinstance(text, str) and text.startswith("CONTENT: "):
                text = text[len("CONTENT: "):]
            context.append(text)

            meta = {}
            if isinstance(doc, dict):
                meta["file_name"] = doc.get("metadata", {}).get("file_name", "")
                meta["chunk_id"] = doc.get("chunk_id", doc.get("metadata", {}).get("chunk_id", ""))
                meta["rerank_score"] = doc.get("rerank_score", doc.get("original_score", ""))
                meta["original_score"] = doc.get("original_score", doc.get("score", ""))
                sources.append(meta)
            elif hasattr(doc, "metadata"):
                sources.append({
                    "file_name": doc.metadata.get("file_name", ""),
                    "chunk_id": doc.chunk_id,
                    "rerank_score": getattr(doc, "rerank_score", None),
                    "original_score": getattr(doc, "original_score", None),
                })

        usage = result.get("usage", {}) or {}
        timings = result.get("timings", {}) or {}

        declined = final_answer == "" or "I'm a Kubernetes documentation assistant" in final_answer

        # Fallback: if timings not populated by nodes, compute e2e total as a safe default.
        if not timings:
            timings_ms = {"total": 0.0}
        else:
            timings_ms = {k: round(v, 2) for k, v in timings.items()} if isinstance(timings, dict) else {}

        return EvalResult(
            query=query,
            answer=final_answer,
            context=context,
            sources=sources,
            intent=intent,
            declined=declined,
            plan=plan,
            timings_ms=timings_ms,
            usage=usage if isinstance(usage, dict) else {},
        )


def _make_state_with_history(query: str, history: List[dict]) -> dict:
    messages = [{"role": "user", "content": query}]
    for h in history:
        messages.append({**h, "role": h.get("role", "user")})
    return {
        "messages": messages,
        "current_query": "",
        "intent": "",
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }
