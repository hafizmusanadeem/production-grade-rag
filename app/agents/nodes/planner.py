import re
from time import perf_counter

import logfire

from app.agents.prompts import build_planner_prompt
from app.agents.state import (
    INTENT_CONVERSATIONAL,
    INTENT_OUT_OF_SCOPE,
    INTENT_TECHNICAL,
    AgentState,
)
from app.exceptions import LLMServiceError
from app.gateway import get_langchain_llm
from app.gateway.client import request_extra_headers

llm = get_langchain_llm(feature = "planner")


_OPERATOR_RE = re.compile(r"\b(?:site|inurl|intitle|intext|filetype):\S+", re.IGNORECASE)


def clean_search_query(query: str, fallback: str) -> str:
    """Cleans the User's Techincal Search Query (removing any links, operators from query) returning from the LLM"""
    cleaned = _OPERATOR_RE.sub("", query).replace('"', "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or fallback


def parse_planner_decision(raw: str, fallback_query: str) -> tuple[str, str]:
    """
    Turn the planner LLM's raw output into (intent, search_query).

    Pure function (no I/O) so the routing contract is unit-testable.
    The search_query is "" unless intent is technical.
    """
    normalized = raw.strip().strip(" .'\"`").upper().replace(" ", "_")
    if normalized == "CONVERSATIONAL":
        return INTENT_CONVERSATIONAL, ""
    if normalized == "OUT_OF_SCOPE":
        return INTENT_OUT_OF_SCOPE, ""
    return INTENT_TECHNICAL, clean_search_query(raw, fallback_query)


def _as_text(content) -> str:
    """LangChain message content is usually str, but can be a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return ""


def planner_node(state: AgentState):
    """
    The Planner classifies the latest message as CONVERSATIONAL, OUT_OF_SCOPE,
    or a technical Kubernetes question (and produces a search query for it).
    """
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = build_planner_prompt(history, user_message)

    with logfire.span("Planner Decision", history_messages=len(state["messages"]) - 1):
        try:
            t0 = perf_counter()
            response = llm.invoke(prompt, extra_headers=request_extra_headers())
            planner_ms = (perf_counter() - t0) * 1000
            raw = _as_text(response.content)
            meta = getattr(response, "usage_metadata", None) or {}
            model_meta = getattr(response, "response_metadata", None) or {}
            usage = {"planner_prompt_tokens": meta.get("input_tokens", 0),
                     "planner_completion_tokens": meta.get("output_tokens", 0),
                     "planner_model": str(model_meta.get("model_name", "unknown") or "unknown")}
        except Exception as exc:
            logfire.exception("Planner LLM call failed")
            raise LLMServiceError(f"Planner LLM call failed: {exc}") from exc

        planner_fallback = not raw.strip()
        if planner_fallback:
            # Degrade explicitly (not silently): treat as technical, search with the
            # raw message. The relevance guard in the retriever still protects us
            # from answering off-topic questions.
            logfire.warn("Planner returned empty output; falling back to raw user message")

        intent, search_query = parse_planner_decision(raw, fallback_query=user_message)
        logfire.info("Intent Identified", intent=intent, search_query=search_query, raw_decision=raw)

    if intent == INTENT_OUT_OF_SCOPE:
        return {
            "intent": INTENT_OUT_OF_SCOPE,
            "current_query": "",
            "status": "Out of scope - not a Kubernetes question",
            "plan": ["Intent: Out of scope", "Retrieval: Skipped"], "timings_ms": {**state.get("timings_ms", {}), "planner": planner_ms}, "usage": {**state.get("usage", {}), **usage}, "sources": [], "declined": False,
        }

    if intent == INTENT_CONVERSATIONAL:
        return {
            "intent": INTENT_CONVERSATIONAL,
            "current_query": "",
            "status": "Handling conversationally (by using memory)",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"], "timings_ms": {**state.get("timings_ms", {}), "planner": planner_ms}, "usage": {**state.get("usage", {}), **usage}, "sources": [], "declined": False,
        }

    plan = ["Intent: Technical", f"Search Term: {search_query}"]
    if planner_fallback:
        plan.append("Planner: fallback to raw question")
    return {
        "intent": INTENT_TECHNICAL,
        "current_query": search_query,
        "status": f"Technical research needed. Searching for: {search_query}\n",
        "plan": plan,
        "timings_ms": {**state.get("timings_ms", {}), "planner": planner_ms},
        "usage": {**state.get("usage", {}), **usage},
        "sources": [],
        "declined": False,
    }
