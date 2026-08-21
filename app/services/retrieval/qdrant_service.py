# qdrant_service.py

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings, validate_env_vars
from app.services.retrieval.embeddings import EmbeddedChunk, embed_query


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_TIMEOUT_SECONDS = 120
QDRANT_UPSERT_BATCH_SIZE = 5
QDRANT_MAX_RETRIES = 3
QDRANT_RETRY_BASE_DELAY_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single search hit returned from Qdrant."""

    chunk_id: str
    page_content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Return the cached Qdrant client.

    The client is created lazily and reused for the lifetime of the process.
    A generous request timeout is configured because embedding uploads can
    contain large HTTP payloads.
    """
    validate_env_vars()

    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    endpoint = settings.QDRANT_CLUSTER_ENDPOINT
    api_key = settings.QDRANT_API_KEY

    if not endpoint:
        raise ValueError("QDRANT_CLUSTER_ENDPOINT is required")

    if not api_key:
        raise ValueError("QDRANT_API_KEY is required")

    _qdrant_client = QdrantClient(
        url=endpoint,
        api_key=api_key,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )

    logfire.info(
        "Qdrant client initialized",
        endpoint=endpoint,
        timeout_seconds=QDRANT_TIMEOUT_SECONDS,
    )

    return _qdrant_client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Ensure that the target Qdrant collection exists and has the expected
    vector dimensionality.

    Raises:
        ValueError: If the existing collection has a different vector size.
    """
    if not collection_name:
        raise ValueError("Qdrant collection name cannot be empty")

    if vector_size <= 0:
        raise ValueError(
            f"Vector size must be positive, got {vector_size}"
        )

    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)

        vectors_config = info.config.params.vectors

        # This service uses a single dense vector configuration.
        if isinstance(vectors_config, qmodels.VectorParams):
            existing_size = vectors_config.size
        else:
            raise ValueError(
                f"Qdrant collection '{collection_name}' does not use "
                "a single dense vector configuration"
            )

        if existing_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{collection_name}' expects vector size "
                f"{existing_size}, but embeddings have size {vector_size}"
            )

        return

    try:
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
            distance="Cosine",
        )

    except Exception:
        # Another worker/process may have created the collection between
        # collection_exists() and create_collection().
        if not client.collection_exists(collection_name):
            raise

        info = client.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        if not isinstance(vectors_config, qmodels.VectorParams):
            raise ValueError(
                f"Qdrant collection '{collection_name}' does not use "
                "a single dense vector configuration"
            )

        if vectors_config.size != vector_size:
            raise ValueError(
                f"Qdrant collection '{collection_name}' expects vector size "
                f"{vectors_config.size}, but embeddings have size {vector_size}"
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_chunks(chunks: list[EmbeddedChunk]) -> int:
    """
    Validate embedding chunks before sending anything to Qdrant.

    Returns:
        The common embedding vector dimension.
    """
    if not chunks:
        return 0

    first_vector = chunks[0].embedding

    if not first_vector:
        raise ValueError(
            f"Chunk '{chunks[0].chunk_id}' has an empty embedding"
        )

    vector_size = len(first_vector)

    for index, chunk in enumerate(chunks):
        if not chunk.chunk_id:
            raise ValueError(
                f"Chunk at index {index} has no chunk_id"
            )

        vector = chunk.embedding

        if not vector:
            raise ValueError(
                f"Chunk '{chunk.chunk_id}' has an empty embedding"
            )

        if len(vector) != vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for chunk "
                f"'{chunk.chunk_id}': expected {vector_size}, "
                f"got {len(vector)}"
            )

        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(
                f"Embedding for chunk '{chunk.chunk_id}' contains "
                "NaN or infinite values"
            )

    return vector_size


def _build_points(
    chunks: list[EmbeddedChunk],
) -> list[qmodels.PointStruct]:
    """Convert embedded chunks into Qdrant points."""
    return [
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


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _upsert_batch_with_retry(
    client: QdrantClient,
    collection_name: str,
    points: list[qmodels.PointStruct],
    batch_number: int,
    total_batches: int,
) -> None:
    """
    Upsert one batch with bounded exponential-backoff retries.

    Qdrant upserts are intentionally idempotent because chunk_id is used as
    the point ID. Retrying the same batch therefore does not create duplicate
    points.
    """
    last_error: Exception | None = None

    for attempt in range(1, QDRANT_MAX_RETRIES + 1):
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

            logfire.info(
                "Qdrant batch uploaded",
                collection=collection_name,
                batch_number=batch_number,
                total_batches=total_batches,
                point_count=len(points),
                attempt=attempt,
            )

            return

        except Exception as exc:
            last_error = exc

            if attempt >= QDRANT_MAX_RETRIES:
                break

            delay = QDRANT_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

            logfire.warning(
                "Qdrant batch upload failed; retrying",
                collection=collection_name,
                batch_number=batch_number,
                total_batches=total_batches,
                point_count=len(points),
                attempt=attempt,
                max_retries=QDRANT_MAX_RETRIES,
                retry_delay_seconds=delay,
                error=str(exc),
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Failed to upload Qdrant batch {batch_number}/{total_batches} "
        f"after {QDRANT_MAX_RETRIES} attempts"
    ) from last_error


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_embeddings(chunks: list[EmbeddedChunk]) -> int:
    """
    Store embedded chunks in Qdrant using bounded batches.

    The operation is idempotent because each chunk's stable chunk_id is used
    as the Qdrant point ID.

    Returns:
        Number of points successfully upserted.
    """
    if not chunks:
        logfire.info("No embeddings to store in Qdrant")
        return 0

    collection_name = settings.QDRANT_COLLECTION

    if not collection_name:
        raise ValueError(
            "QDRANT_COLLECTION is required to store embeddings"
        )

    vector_size = _validate_chunks(chunks)

    client = get_qdrant_client()

    ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    points = _build_points(chunks)

    total_points = len(points)
    batch_size = QDRANT_UPSERT_BATCH_SIZE
    total_batches = (
        total_points + batch_size - 1
    ) // batch_size

    logfire.info(
        "Starting Qdrant embedding upload",
        collection=collection_name,
        point_count=total_points,
        vector_size=vector_size,
        batch_size=batch_size,
        total_batches=total_batches,
    )

    for start in range(0, total_points, batch_size):
        batch = points[start:start + batch_size]
        batch_number = (start // batch_size) + 1

        _upsert_batch_with_retry(
            client=client,
            collection_name=collection_name,
            points=batch,
            batch_number=batch_number,
            total_batches=total_batches,
        )

    logfire.info(
        "Stored embeddings in Qdrant",
        collection=collection_name,
        point_count=total_points,
        vector_size=vector_size,
        batch_size=batch_size,
    )

    return total_points


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@logfire.instrument(
    "Search Qdrant {query=}",
    extract_args=("query",),
)
def search_qdrant(
    query: str,
    limit_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Search Qdrant using an embedded natural-language query.

    Args:
        query: User's natural-language search query.
        limit_k: Maximum number of chunks to retrieve.

    Returns:
        Ranked list of RetrievedChunk objects.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    if limit_k <= 0:
        raise ValueError(
            f"limit_k must be greater than zero, got {limit_k}"
        )

    collection_name = settings.QDRANT_COLLECTION

    if not collection_name:
        raise ValueError(
            "QDRANT_COLLECTION is required to query Qdrant"
        )

    client = get_qdrant_client()

    try:
        query_vector = embed_query(query)

        if not query_vector:
            raise ValueError(
                "Embedding model returned an empty query vector"
            )

        response = client.query_points(
            collection_name=collection_name,
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
                    page_content=str(page_content),
                    score=float(point.score),
                    metadata=payload,
                )
            )

        logfire.info(
            "Qdrant search complete",
            collection=collection_name,
            result_count=len(retrieved),
            limit_k=limit_k,
        )

        return retrieved

    except Exception as exc:
        logfire.error(
            "Qdrant search failed",
            collection=collection_name,
            limit_k=limit_k,
            error=str(exc),
        )
        raise