"""app/generator.py -- pure generator functions.

Used by:
  - app/agents/nodes/responder.py (generate_node wrapper)
  - evals/eval_files/eval_generator.py (isolated faithfulness + relevancy testing)

The reference project's generate() pulls from src.generator; here we mirror that
contract so both caller paths work without duplication. The prompt builders are
pure (no I/O), so P8 can hash them for regression detection.
"""
import logfire

from app.gateway import portkey_client, extract_cache_status
from app.config import settings

# Pure prompt builders (no embedding/search/portkey side effects) -- hashable.

def build_prompt(query: str, context: list[str], intent: str = "technical",
                 history: list[dict] | None = None) -> str:
    history_str = ""
    for msg in history or []:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    user_msg = (history[-1]["content"] if history and isinstance(history[-1], dict)
                else "") or query

    if intent == "conversational":
        return (f"""You are a friendly and helpful Kubernetes documentation assistant.
Answer the user's latest message using ONLY the CONVERSATION HISTORY below.
If the message asks for something unrelated to Kubernetes that the history does not cover, politely say you can only help with Kubernetes questions.

CONVERSATION HISTORY:
{history_str}

LATEST MESSAGE:
"{user_msg}"
""")
    else:
        full_context = "\n\n".join(context)
        return (f"""You are a Senior Kubernetes Technical Architect.
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
""")


def generate_answer(query: str, documents: list[str], intent: str = "technical",
                    history: list[dict] | None = None) -> str:
    """Run generation through the real Portkey-backed LLM."""
    if isinstance(documents, list):
        text_docs = [d if isinstance(d, str) else d.page_content if hasattr(d, "page_content") else str(d) for d in documents]
    else:
        text_docs = [documents]

    prompt = build_prompt(query, text_docs, intent=intent, history=history)

    try:
        response = portkey_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        choices = getattr(response, "choices", None)
        content = choices[0].message.content if choices else ""
        if not content or not content.strip():
            return ""
        return content
    except Exception as exc:
        logfire.exception("Generator failed", error=str(exc))
        return ""


def generate(query: str, documents, intent: str = "technical",
             history: list[dict] | None = None) -> str:
    """Public alias matching the reference `generate(query, context) -> answer`."""
    return generate_answer(query, documents, intent=intent, history=history)
