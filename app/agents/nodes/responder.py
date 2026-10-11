import logfire

from app.gateway import portkey_client, extract_cache_status
from app.agents.state import AgentState, INTENT_CONVERSATIONAL
from app.exceptions import EmptyLLMResponseError, LLMServiceError


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

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if is_conversational:
        logfire.info("Generating conversational response using memory.", history_message=len(state["messages"]) - 1)
        prompt = f"""
        You are a friendly and helpful Kubernetes documentation assistant.
        Answer the user's latest message using ONLY the CONVERSATION HISTORY below.
        If the message asks for something unrelated to Kubernetes that the history does
        not cover, politely say you can only help with Kubernetes questions.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        max_content_chars = 25000
        full_context = ""
        used_docs = 0

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_content_chars:
                full_context += doc + "\n\n"
                used_docs += 1
            else:
                logfire.warn("Context truncated to fit Groq TPM limits.", used_docs=used_docs, total_docs=len(state["documents"]), max_chars=max_content_chars)
                break

        logfire.info("Generating Technical RAG response", used_docs=used_docs, context_chars=len(full_context))
        prompt = f"""
        You are a Senior Kubernetes Technical Architect.
        Answer the question using ONLY the TECHNICAL CONTEXT provided below.
        Do not use outside knowledge. If the context does not contain the answer, say
        that you could not find it in the documentation instead of guessing.
        If the question is not about Kubernetes, say you can only help with Kubernetes questions.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("LLM Synthesis", mode="conversational" if is_conversational else "rag", prompt_chars=len(prompt)) as span:
        try:
            response = portkey_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1
            )
        except Exception as exc:
            logfire.exception("LLM generation failed")
            raise LLMServiceError(f"LLM generation failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        content = choices[0].message.content if choices else None
        if not content or not content.strip():
            raise EmptyLLMResponseError(
                f"LLM returned no content (model={getattr(response, 'model', 'unknown')})"
            )

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

        return {
            "final_answer": content,
            "status": "Cache Hit - instant response" if is_cache_hit else "Response generated",
            "plan": state["plan"] + ["Cache Hit"] if is_cache_hit else state["plan"],
            "messages": [{"role": "assistant", "content": content}]
        }
