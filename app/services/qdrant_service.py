import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embeddings.embeddings import EmbeddedChunk
from typing import List

# Initialize QdrantClient
qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key = settings.QDRANT_API_KEY
)

def search_qdrant(query: str, limit_k: int = 5) -> List[EmbeddedChunk]:
    """
    Search Qdrant for the most relevant chunks based on the query.
    """
    query_vector = embed_query(query)
    results = qdrant_client.query_points(
        collection_name="production_rag",
        query=query_vector,
        limit=limit_k,
        with_payload=True
    )

    results = []

    for result in results.points:
        result.append({
            "chunk_id": result.id,
            "page_content": result.payload.get("text", ""),
            "embedding": result.vector,
            "metadata": result.payload
        })

    return [EmbeddedChunk.from_qdrant_result(result) for result in results]