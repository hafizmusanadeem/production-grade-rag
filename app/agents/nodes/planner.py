import logfire
import re

from app.gateway import get_langchain_llm
from app.agents.state import AgentState


llm = get_langchain_llm(feature = "planner")


_OPERATOR_RE = re.compile(r"\b(?:site|inurl|intitle|intext|filetype):\S+", re.IGNORECASE)
def clean_search_query(query: str, fallback: str) -> str:
    """Cleans the User's Techincal Search Query (removing any links, operators from query) returning from the LLM"""
    cleaned = _OPERATOR_RE.sub("", query).replace('"', "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or fallback
def planner_node(state: AgentState):
    """
    The Planner determines if the search is needed based on the entire conversation.
    """
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query written as plain natural language: only keywords or a short phrase. Do NOT use search operators (site:, inurl:, filetype:), quotation marks, or boolean words.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("Planner Decision", history_messages=len(state["messages"]) -1):
        decision = llm.invoke(prompt).content.strip()
        if decision != "CONVERSATIONAL":
            decision = clean_search_query(decision, fallback = user_message)
        logfire.info("Intent Identified", decision=decision, is_conversational=decision == "CONVERSATIONAL")

    if decision.strip(" .'\"").upper() == "CONVERSATIONAL":
        return {
            "current_query":"CONVERSATIONAL",
            "status":"Handling conversationally (by using memory)",
            "plan":["Intent: Conversational/Memory", "Retrieval:Skipped"]
        }

    return {
        "current_query":decision,
        "status":f"Technical research needed. Searching for: {decision}\n",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }