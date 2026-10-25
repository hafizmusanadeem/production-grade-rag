"""
Canned refusals. No LLM call, so a decline is instant, free, deterministic,
and cannot be talked out of by the user's wording.
"""
import logfire

from app.agents.state import INTENT_OUT_OF_SCOPE, AgentState

OUT_OF_SCOPE_MESSAGE = (
    "I'm a Kubernetes documentation assistant, so I can only help with Kubernetes "
    "questions (for example Jobs and CronJobs, Pods, Deployments, networking, storage, "
    "RBAC, or autoscaling). I can't answer questions outside that scope. "
    "Feel free to ask me something about Kubernetes!"
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything relevant to that in the ingested Kubernetes documentation, "
    "so I'd rather not guess. Try rephrasing the question, or check that the relevant "
    "document has been ingested."
)


def decline_node(state: AgentState):
    if state.get("intent") == INTENT_OUT_OF_SCOPE:
        message, step = OUT_OF_SCOPE_MESSAGE, "Declined: out of scope"
    else:
        message, step = NO_CONTEXT_MESSAGE, "Declined: no relevant context"

    logfire.info("Declining request", reason=step, intent=state.get("intent"))

    return {
        "final_answer": message,
        "status": step,
        "plan": state["plan"] + [step],
        # Keep the exchange in thread memory so follow-ups have context.
        "messages": [{"role": "assistant", "content": message}],
        "declined": True, "timings_ms": {**state.get("timings_ms", {})},
        "usage": {**state.get("usage", {})}, "sources": [],
    }
