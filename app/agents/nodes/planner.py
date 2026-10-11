import logfire
import re

from app.gateway import get_langchain_llm
from app.agents.state import (
    AgentState,
    INTENT_CONVERSATIONAL,
    INTENT_OUT_OF_SCOPE,
    INTENT_TECHNICAL,
)
from app.exceptions import LLMServiceError


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

    prompt = f"""
    You are the intake Planner for a Kubernetes documentation assistant.
    The assistant ONLY answers questions about Kubernetes and closely related
    cloud-native topics (containers, Pods, Jobs/CronJobs, Deployments, networking,
    storage, RBAC, autoscaling, Helm, monitoring such as Prometheus, and similar).
    Messages may be written in any language, including Roman Urdu or Hinglish.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Classify the latest message and output exactly ONE of the following:

    1. CONVERSATIONAL
       The message is a greeting/thanks, or can be answered using ONLY the
       conversation history above (e.g., "what is my name", "summarize what you just said").

    2. OUT_OF_SCOPE
       The message is not about Kubernetes or related cloud-native tooling and is not
       answerable from the history. Examples: geography, politics, sports, general
       trivia, personal advice, writing poems, unrelated programming help.

    3. A search query
       The message is a Kubernetes/cloud-native question that needs documentation.
       Output a refined search query written as plain natural language: only keywords
       or a short phrase. Do NOT use search operators (site:, inurl:, filetype:),
       quotation marks, or boolean words.

    Output ONLY 'CONVERSATIONAL', 'OUT_OF_SCOPE', or the search query. No explanation.
    """

    with logfire.span("Planner Decision", history_messages=len(state["messages"]) - 1):
        try:
            raw = _as_text(llm.invoke(prompt).content)
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
            "plan": ["Intent: Out of scope", "Retrieval: Skipped"],
        }

    if intent == INTENT_CONVERSATIONAL:
        return {
            "intent": INTENT_CONVERSATIONAL,
            "current_query": "",
            "status": "Handling conversationally (by using memory)",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    plan = ["Intent: Technical", f"Search Term: {search_query}"]
    if planner_fallback:
        plan.append("Planner: fallback to raw question")
    return {
        "intent": INTENT_TECHNICAL,
        "current_query": search_query,
        "status": f"Technical research needed. Searching for: {search_query}\n",
        "plan": plan,
    }
