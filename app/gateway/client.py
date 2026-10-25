from contextvars import ContextVar

from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, Portkey, createHeaders

from app.config import settings

_bypass_cache = ContextVar("bypass_cache", default=False)


def set_bypass_cache(value: bool):
    return _bypass_cache.set(bool(value))


def reset_bypass_cache(token):
    _bypass_cache.reset(token)


def request_extra_headers():
    return {"x-portkey-cache-force-refresh": "true"} if _bypass_cache.get() else {}


# Production gateway config:
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @brag/llama-3.1-8b-instant on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target

# GATEWAY_CONFIG = {
#     "strategy": {"mode": "fallback"},
#     "cache": {"mode": "simple"},
#     "retry": {
#         "attempts": 2,
#         "on_status_codes": [429, 503]
#     },
#     "targets": [
#         {"override_params": {"model": f"@{settings.PORTKEY_SLUG_1}/openrouter/free"}},
#         {"override_params": {"model": f"@{settings.PORTKEY_SLUG_2}/openrouter/free"}},
#     ]
# }

# The upper defined GATEWAY_CONFIG is setup in portkey.com/config from where It's imported.

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config= settings.PORTKEY_CONFIG_ID # Portkey Defined Config
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.

    timeout / max_retries bound how long a request can hang on a slow provider, so a
    stuck LLM call surfaces as an error (HTTP 502) instead of an endless spinner.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.PORTKEY_SLUG_2}/openrouter/free",
        temperature=0,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
