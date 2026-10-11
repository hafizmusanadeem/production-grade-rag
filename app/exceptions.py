"""
Typed application errors.

Every failure that can reach the API boundary is a RAGError (or an unexpected
Exception, which the catch-all handler turns into a generic 500).

Each RAGError carries:
  status_code     -> the HTTP status the API returns
  code            -> a stable machine-readable string clients can branch on
  public_message  -> safe, user-facing text (never contains internals)
  detail          -> the internal explanation; logged, never returned to clients

Raise these with `raise XError("what actually went wrong") from original_exc`
so the cause chain is preserved in Logfire while the client only sees
`public_message`.
"""


class RAGError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    public_message: str = "Something went wrong while processing your request."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.public_message
        super().__init__(self.detail)


class RetrievalError(RAGError):
    """Vector search (query embedding or Qdrant) failed."""

    status_code = 503
    code = "retrieval_unavailable"
    public_message = (
        "The knowledge base is temporarily unavailable. Please try again shortly."
    )


class RerankError(RAGError):
    """The cross-encoder reranker failed."""

    status_code = 503
    code = "rerank_failed"
    public_message = (
        "Search results could not be ranked right now. Please try again shortly."
    )


class LLMServiceError(RAGError):
    """The LLM gateway call failed (network, auth, rate limit, provider error)."""

    status_code = 502
    code = "llm_unavailable"
    public_message = (
        "The language model is temporarily unavailable. Please try again shortly."
    )


class EmptyLLMResponseError(LLMServiceError):
    """The LLM call succeeded but returned no usable text."""

    code = "llm_empty_response"
    public_message = (
        "The language model returned an empty response. Please try again."
    )
