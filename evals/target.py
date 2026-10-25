"""evals/target.py -- the eval target adapter.

A single object that runs the real LangGraph agent and returns everything the
eval suite needs: the answer, retrieved context (chunk text with the "CONTENT:"
prefix stripped), sources metadata, intent, plan, timings, token usage, error
code, and whether the agent declined. This is the seam every eval hits.
"""
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional

import logfire

from app.agents.graph import rag_agent
from app.agents.nodes.decline import NO_CONTEXT_MESSAGE, OUT_OF_SCOPE_MESSAGE
from app.exceptions import RAGError
from app.gateway import reset_bypass_cache, set_bypass_cache


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
        self.bypass_cache = bypass_cache
        self.results: list[EvalResult] = []

    def invoke(self, query: str, *, thread_id: Optional[str] = None,
               history: Optional[List[dict]] = None) -> EvalResult:
        result = self._invoke(query, thread_id=thread_id, history=history)
        self.results.append(result)
        return result

    def _invoke(self, query: str, *, thread_id: Optional[str] = None,
                history: Optional[List[dict]] = None) -> EvalResult:
        tid = thread_id or f"eval-{uuid.uuid4().hex}"
        state = _make_state_with_history(query, history or []) if history else _make_initial_state(query)
        started = perf_counter()
        cache_token = set_bypass_cache(self.bypass_cache)
        try:
            result = rag_agent.invoke(state, config={"configurable": {"thread_id": tid}})
        except RAGError as exc:
            return EvalResult(query=query, answer="", context=[], error=exc.code,
                              timings_ms={"total": (perf_counter() - started) * 1000})
        except Exception as exc:
            logfire.error("AgentTarget invoke failed", error_id=uuid.uuid4().hex[:12],
                          error=str(exc), _exc_info=exc)
            return EvalResult(query=query, answer="", context=[], error="internal_error",
                              timings_ms={"total": (perf_counter() - started) * 1000})
        finally:
            reset_bypass_cache(cache_token)
        wall_ms = (perf_counter() - started) * 1000
        answer = result.get("final_answer", "") or ""
        context = []
        for doc in result.get("documents", []) or []:
            if isinstance(doc, str):
                context.append(doc.removeprefix("CONTENT: "))
        timings = dict(result.get("timings_ms", {}) or {})
        timings["total"] = wall_ms
        declined = result.get("declined")
        if declined is None:
            declined = answer in (OUT_OF_SCOPE_MESSAGE, NO_CONTEXT_MESSAGE)
        return EvalResult(
            query=query, answer=answer, context=context,
            sources=list(result.get("sources", []) or []),
            intent=result.get("intent", "unknown"), declined=bool(declined),
            plan=result.get("plan", []) or [], timings_ms=timings,
            usage=dict(result.get("usage", {}) or {}),
        )



def _make_state_with_history(query: str, history: List[dict]) -> dict:
    messages = []
    for item in history:
        if isinstance(item, dict) and item.get("role", "user") in {"user", "assistant", "system"}:
            if isinstance(item.get("content", ""), str):
                messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    messages.append({"role": "user", "content": query})
    return {"messages": messages, "current_query": "", "intent": "", "documents": [],
            "plan": [], "status": "", "final_answer": "", "timings_ms": {},
            "usage": {}, "sources": [], "declined": False}
