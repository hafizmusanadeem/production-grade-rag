#qdrant_service
from typing import Any
from dataclasses import dataclass, field

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.services.retrieval.embeddings import EmbeddedChunk, embed_query


@dataclass
class RetrievedChunk:
    """A single search hit: the chunk text, its similarity score, and metadata."""

    chunk_id: str
    page_content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return the cached Qdrant client, creating it lazily if necessary."""
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    if not settings.QDRANT_CLUSTER_ENDPOINT:
        raise ValueError(
            "QDRANT_CLUSTER_ENDPOINT is required to connect to Qdrant"
        )

    if not settings.QDRANT_API_KEY:
        raise ValueError(
            "QDRANT_API_KEY is required to connect to Qdrant"
        )

    _qdrant_client = QdrantClient(
        url=settings.QDRANT_CLUSTER_ENDPOINT,
        api_key=settings.QDRANT_API_KEY,
    )

    return _qdrant_client

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)

        existing_size = info.config.params.vectors.size

        if existing_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{collection_name}' expects vector size "
                f"{existing_size}, but embeddings have size {vector_size}"
            )

        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )

    logfire.info(
        "Created Qdrant collection",
        collection=collection_name,
        vector_size=vector_size,
    )

def store_embeddings(chunks: list[EmbeddedChunk]) -> int:
    if not chunks:
        return 0

    collection_name = settings.QDRANT_COLLECTION

    if not collection_name:
        raise ValueError(
            "QDRANT_COLLECTION is required to store embeddings"
        )

    client = get_qdrant_client()

    vector_size = len(chunks[0].embedding)

    ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    points = [
        qmodels.PointStruct(
            id=chunk.chunk_id,
            vector=chunk.embedding,
            payload={
                "page_content": chunk.page_content,
                **chunk.metadata,
            },
        )
        for chunk in chunks
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    logfire.info(
        "Stored embeddings in Qdrant",
        collection=collection_name,
        point_count=len(points),
        vector_size=vector_size,
    )

    return len(points)

@logfire.instrument(
    "Search Qdrant {query=}",
    extract_args=("query",),
)
def search_qdrant(
    query: str,
    limit_k: int = 5,
) -> list[RetrievedChunk]:

    if not settings.QDRANT_COLLECTION:
        raise ValueError(
            "QDRANT_COLLECTION is required to query Qdrant"
        )

    client = get_qdrant_client()

    query_vector = embed_query(query)

    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit_k,
        with_payload=True,
    )

    retrieved: list[RetrievedChunk] = []

    for point in response.points:
        payload = dict(point.payload or {})

        page_content = payload.pop(
            "page_content",
            "",
        )

        retrieved.append(
            RetrievedChunk(
                chunk_id=str(point.id),
                page_content=page_content,
                score=point.score,
                metadata=payload,
            )
        )

    logfire.info(
        "Qdrant search complete",
        collection=settings.QDRANT_COLLECTION,
        query=query,
        result_count=len(retrieved),
    )

    return retrieved