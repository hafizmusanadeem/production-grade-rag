from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import (
    AgentState,
    INTENT_CONVERSATIONAL,
    INTENT_OUT_OF_SCOPE,
)
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.decline import decline_node

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("decline", decline_node)


def route_planner(state: AgentState):
    """
    Route the workflow based on the planner's decision:
      out of scope   -> decline (canned refusal, no retrieval, no LLM)
      conversational -> responder (answers from thread memory)
      technical      -> retriever
    """
    intent = state.get("intent")
    if intent == INTENT_OUT_OF_SCOPE:
        return "decline"
    if intent == INTENT_CONVERSATIONAL:
        return "responder"
    return "retriever"


def route_retriever(state: AgentState):
    """If nothing relevant was retrieved, decline rather than let the LLM answer ungrounded."""
    return "responder" if state.get("documents") else "decline"


workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
        "decline": "decline",
    }
)

workflow.add_conditional_edges(
    "retriever",
    route_retriever,
    {
        "responder": "responder",
        "decline": "decline",
    }
)

workflow.add_edge("responder", END)
workflow.add_edge("decline", END)

checkpointer = MemorySaver()

rag_agent = workflow.compile(checkpointer = checkpointer)
