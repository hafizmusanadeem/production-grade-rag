import logfire

from app.agents.state import AgentState
from app.config import settings
from app.exceptions import RAGError, RerankError, RetrievalError
from app.services.retrieval.qdrant_service import search_qdrant
from app.services.retrieval.reranker import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.

    Failure handling:
      - Qdrant / embedding failure  -> RetrievalError (HTTP 503)
      - Reranker failure            -> RerankError    (HTTP 503)
      - Nothing relevant found      -> returns documents=[]; the graph routes to the
                                       decline node instead of letting the LLM guess.
    """

    query = state["current_query"]

    with logfire.span("Knowledge Retrieval"):
        logfire.info("Searching Qdrant for", query=query)
        try:
            raw_results = search_qdrant(query=query, limit_k=15)
        except RAGError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        logfire.info("Retrieved candidates from Vector DB", candidate_count=len(raw_results), vector_db="Qdrant")

        if not raw_results:
            logfire.warn("No candidates returned from Qdrant", query=query)
            return {
                "documents": [],
                "status": "No documents found.",
                "plan": state["plan"] + ["Context: none found"],
            }

        with logfire.span("Semantic Reranking"):
            try:
                reranked = rerank_documents(query=query, documents=raw_results, top_n=5)
            except RAGError:
                raise
            except Exception as exc:
                raise RerankError(f"Reranking failed: {exc}") from exc

            logfire.info("Reranking complete", reranked_count=len(reranked))

        top_score = max((doc.rerank_score for doc in reranked), default=None)
        relevant = [doc for doc in reranked if doc.rerank_score >= settings.RERANK_MIN_SCORE]

        # Log the score so RERANK_MIN_SCORE can be calibrated from real traffic.
        logfire.info(
            "Relevance filter applied",
            top_rerank_score=top_score,
            threshold=settings.RERANK_MIN_SCORE,
            kept=len(relevant),
            dropped=len(reranked) - len(relevant),
        )

        if not relevant:
            logfire.warn(
                "No chunk cleared the relevance threshold; declining instead of answering ungrounded",
                query=query,
                top_rerank_score=top_score,
                threshold=settings.RERANK_MIN_SCORE,
            )
            return {
                "documents": [],
                "status": "No relevant documents found.",
                "plan": state["plan"] + ["Context: none relevant"],
            }

        formatted_docs = [f"CONTENT: {doc.page_content}" for doc in relevant]

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
