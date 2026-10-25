"""app/generator.py -- pure generator functions.

Used by:
  - app/agents/nodes/responder.py (generate_node wrapper)
  - evals/eval_files/eval_generator.py (isolated faithfulness + relevancy testing)
"""
import logfire

from app.agents.prompts import build_prompt as _build_prompt
from app.gateway import portkey_client


def build_prompt(query, context, intent="technical", history=None):
    """Delegate to the single hardened prompt builder (app/agents/prompts.py)."""
    return _build_prompt(query, context, intent=intent, history=history)


def format_context(chunks):
    output = []
    for chunk in chunks or []:
        text = chunk if isinstance(chunk, str) else getattr(chunk, "page_content", str(chunk))
        output.append(text if text.startswith("CONTENT: ") else f"CONTENT: {text}")
    return output


def generate_answer(query: str, documents, intent: str = "technical",
                    history: list[dict] | None = None) -> str:
    """Run generation through the real Portkey-backed LLM."""
    if isinstance(documents, list):
        text_docs = [d if isinstance(d, str)
                     else d.page_content if hasattr(d, "page_content") else str(d)
                     for d in documents]
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
            raise ValueError("Generator returned an empty response")
        return content
    except Exception as exc:
        logfire.exception("Generator failed", error=str(exc))
        raise


def generate(query: str, documents, intent: str = "technical",
             history: list[dict] | None = None) -> str:
    """Public alias matching the reference `generate(query, context) -> answer`."""
    return generate_answer(query, documents, intent=intent, history=history)
