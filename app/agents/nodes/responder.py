from time import perf_counter

import logfire

from app.agents.prompts import build_prompt
from app.agents.state import INTENT_CONVERSATIONAL, AgentState
from app.exceptions import EmptyLLMResponseError, LLMServiceError
from app.gateway import extract_cache_status, portkey_client, request_extra_headers
from app.guardrails import output_guard


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.

    Failure handling:
      - Gateway/provider error -> LLMServiceError      (HTTP 502)
      - Empty completion       -> EmptyLLMResponseError (HTTP 502)
    """

    is_conversational = state.get("intent") == INTENT_CONVERSATIONAL

    history = state["messages"][:-1]
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    if is_conversational:
        used_docs = 0
        prompt = build_prompt(user_msg, [], intent="conversational", history=history)
    else:
        selected_docs = []
        used_docs = 0
        chars = 0
        for doc in state["documents"]:
            if chars + len(doc) < 25000:
                selected_docs.append(doc)
                chars += len(doc)
                used_docs += 1
            else:
                logfire.warn("Context truncated to fit gateway limits.", used_docs=used_docs,
                             total_docs=len(state["documents"]), max_chars=25000)
                break
        prompt = build_prompt(user_msg, selected_docs, intent="technical", history=history)

    with logfire.span("LLM Synthesis", mode="conversational" if is_conversational else "rag", prompt_chars=len(prompt)) as span:
        try:
            t0 = perf_counter()
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                extra_headers=request_extra_headers(),
            )
            generation_ms = (perf_counter() - t0) * 1000
        except Exception as exc:
            logfire.exception("LLM generation failed")
            raise LLMServiceError(f"LLM generation failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        content = choices[0].message.content if choices else None
        if not content or not content.strip():
            raise EmptyLLMResponseError(
                f"LLM returned no content (model={getattr(response, 'model', 'unknown')})"
            )

        guarded = output_guard(content)
        content = guarded.answer
        if guarded.reasons:
            logfire.warn("Output guardrail modified response", reasons=guarded.reasons)
        cache_status = extract_cache_status(response)
        usage = getattr(response, "usage", None)

        if usage is not None:
            span.set_attribute("prompt_tokens", usage.prompt_tokens)
            span.set_attribute("completion_tokens", usage.completion_tokens)
        span.set_attribute("model", getattr(response, "model", "unknown"))
        span.set_attribute("cache_status", cache_status)
        span.set_attribute("answer_chars", len(content))

        is_cache_hit = cache_status == "HIT"

        if is_cache_hit:
            logfire.info("Gateway Cache Hit - response served from PortKey cache.")
        else:
            logfire.info("Response synthesized via LLM.")

        new_usage = {**state.get("usage", {}),
                     "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage is not None else 0,
                     "completion_tokens": getattr(usage, "completion_tokens", 0) if usage is not None else 0,
                     "cache_status": cache_status,
                     "responder_model": str(getattr(response, "model", None) or "unknown"),
                     "context_docs_used": used_docs if not is_conversational else 0}
        plan = state["plan"] + (["Cache Hit"] if is_cache_hit else [])
        if guarded.reasons:
            plan.append("Guardrail: output modified")
        return {"final_answer": content,
                "status": "Cache Hit - instant response" if is_cache_hit else "Response generated",
                "plan": plan, "messages": [{"role": "assistant", "content": content}],
                "timings_ms": {**state.get("timings_ms", {}), "generate": generation_ms},
                "usage": new_usage, "declined": guarded.blocked}
