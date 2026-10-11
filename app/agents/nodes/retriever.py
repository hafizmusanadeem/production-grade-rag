import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_qdrant
from app.services.retrieval.reranker import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """

    query = state["current_query"]

    with logfire.span("Knowledge Retrieval"):
        logfire.info("Searching Qdrant for", query=query)
        raw_results = search_qdrant(query= query, limit_k= 15)
        logfire.info("Retrieved candidates from Vector DB",candidate_count=len(raw_results),vector_db="Qdrant")

        with logfire.span("Semantic Reranking"):
            reranked_contents = rerank_documents(query= query, documents= raw_results, top_n= 5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks", reranked_count=len(reranked_contents))

        formatted_docs = [f"CONTENT: {doc.page_content}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }