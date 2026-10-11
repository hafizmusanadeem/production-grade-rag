from typing import TypedDict, List, Annotated
import operator

# Planner intents. The graph routes on these, not on magic strings in current_query.
INTENT_TECHNICAL = "technical"
INTENT_CONVERSATIONAL = "conversational"
INTENT_OUT_OF_SCOPE = "out_of_scope"


class AgentState(TypedDict):
    """
    Using Annotated with operator.add ensures that messages are appended to the history rather than replaced.
    """
    messages: Annotated[List[dict], operator.add]
    current_query: str   # search query; only meaningful when intent == technical
    intent: str          # one of the INTENT_* constants
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
