from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.ingestion.chunking.chunker import Chunk


@dataclass
class EmbeddedChunk:
    chunk_id: str
    page_content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required to generate embeddings")

    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )


@logfire.instrument("Generate embeddings for chunks", extract_args=False)
def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    if not chunks:
        logfire.warn("No chunks provided for embedding")
        return []

    client = _build_embeddings_client()
    texts = [chunk.page_content for chunk in chunks]

    logfire.info(
        "Embedding chunks",
        chunk_count=len(chunks),
        model=settings.GEMINI_EMBEDDING_MODEL,
    )
    vectors = client.embed_documents(texts)

    embedded_chunks = [
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            page_content=chunk.page_content,
            embedding=vector,
            metadata=chunk.metadata,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    logfire.info(
        "Embeddings generated",
        chunk_count=len(embedded_chunks),
        vector_dimension=len(embedded_chunks[0].embedding),
    )
    return embedded_chunks


@logfire.instrument("Embed query", extract_args=("query",))
def embed_query(query: str) -> list[float]:
    """
    Embed a single search query for use against vectors stored by embed_chunks().

    Uses the same Gemini embedding model/client as ingestion so query and
    document vectors live in the same space. GoogleGenerativeAIEmbeddings
    internally tags this as a "retrieval_query" embedding (vs. "retrieval_document"
    for embed_documents), which is the correct asymmetric mode for RAG search.
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty")

    client = _build_embeddings_client()
    vector = client.embed_query(query)

    logfire.info(
        "Query embedded",
        model=settings.GEMINI_EMBEDDING_MODEL,
        vector_dimension=len(vector),
    )
    return vector