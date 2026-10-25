import logfire

from app.agents.state import AgentState
from app.config import RERANK_TOP_N, RETRIEVAL_LIMIT_K, settings
from app.services.retrieval.retrieval_pipeline import QueryRequest, RetrievalPipeline

pipeline = RetrievalPipeline()


def _build_sources(chunks):
    """Per-chunk source metadata for the eval suite and the /query response.

    FlashRank returns numpy.float32 scores, which LangGraph's MemorySaver cannot
    msgpack-serialise when checkpointing state -- coerce to plain floats.
    """
    return [{"file_name": r.metadata.get("file_name", ""), "chunk_id": r.chunk_id,
             "rerank_score": float(r.rerank_score), "original_score": float(r.original_score)}
            for r in chunks]


def retrieve_node(state: AgentState):
    query = state["current_query"]
    req = QueryRequest(query=query, limit_k=RETRIEVAL_LIMIT_K, top_n=RERANK_TOP_N,
                       use_reranker=True, min_rerank_score=settings.RERANK_MIN_SCORE)
    with logfire.span("Knowledge Retrieval"):
        response = pipeline.process_query(req)
        logfire.info("Relevance filter applied", top_rerank_score=response.top_rerank_score,
                     threshold=settings.RERANK_MIN_SCORE, kept=len(response.results),
                     dropped=max(response.retrieved_count - len(response.results), 0))
    sources = _build_sources(response.results)
    timings = {**state.get("timings_ms", {}), **response.timings_ms}
    if response.retrieved_count == 0:
        return {"documents": [], "sources": [], "timings_ms": timings,
                "status": "No documents found.", "plan": state["plan"] + ["Context: none found"]}
    if not response.results:
        logfire.warn("No chunk cleared the relevance threshold; declining instead of answering ungrounded",
                     query=query, top_rerank_score=response.top_rerank_score,
                     threshold=settings.RERANK_MIN_SCORE)
        return {"documents": [], "sources": [], "timings_ms": timings,
                "status": "No relevant documents found.", "plan": state["plan"] + ["Context: none relevant"]}
    docs = [f"CONTENT: {r.page_content}" for r in response.results]
    return {"documents": docs, "sources": sources, "timings_ms": timings,
            "status": "Found technical context.", "plan": state["plan"] + ["Context Retrieved"]}
