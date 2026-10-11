import logfire

from app.gateway import portkey_client, extract_cache_status
from app.agents.state import AgentState


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """

    is_conversational = state["current_query"] == "CONVERSATIONAL"

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if is_conversational:
        logfire.info("Generating conversational response using memory.", history_message=len(state["messages"]) - 1)
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

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
                logfire.warning("Context truncated to fit Groq TPM limits.", used_docs=used_docs, total_docs=len(state["documents"]), max_chars=max_content_chars)
                break

        logfire.info("Generating Technical RAG response", used_docs=used_docs, context_chars=len(full_context))
        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

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

            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            usage = getattr(response, "usage", None)

            if usage is not None:    
                span.set_attribute("model", usage.model)
                span.set_attribute("prompt_tokens", usage.prompt_tokens)
                span.set_attribute("completion_tokens", usage.completion_tokens)
            span.set_attribute("cache_status", cache_status)
            span.set_attribute("answer_chars", len(content or ""))

            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("Gateway Cache Hit - response served from PortKey cache.")
                plan_update = state["plan"] + ["Cache: Hit"]
                status = "Cache Hit - instant response"
            else:
                logfire.info("Response synthesized via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": "Cache Hit - instant response" if is_cache_hit else "Response generated",
                "plan": state["plan"] + ["Cache Hit"] if is_cache_hit else state["plan"],
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e