"""
Retrieval Pipeline for RAG System.

This module orchestrates the complete retrieval flow:
1. Accepts a validated QueryRequest object
2. Searches Qdrant for relevant chunks
3. Reranks the retrieved results using FlashRank cross-encoder
4. Returns the final reranked results in a PipelineResponse

Usage:
    from app.services.retrieval.retrieval_pipeline import RetrievalPipeline, QueryRequest

    pipeline = RetrievalPipeline()
    request = QueryRequest(query="What is the capital of France?", limit_k=5, top_n=3)
    result = pipeline.process_query(request)
"""

import json
import logfire
from dataclasses import dataclass, asdict, field
from typing import Any
from time import perf_counter

from app.services.retrieval.qdrant_service import search_qdrant, RetrievedChunk
from app.services.retrieval.reranker import rerank_documents


@dataclass
class QueryRequest:
    """Input request model for the retrieval pipeline."""
    query: str
    limit_k: int = 5          # Number of chunks to retrieve from Qdrant
    top_n: int = 5            # Number of results after reranking

    def __post_init__(self):
        """Validate fields upon instantiation (Fail Fast)."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty or whitespace")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryRequest":
        """Create QueryRequest from a dictionary."""
        return cls(
            query=data.get("query", ""),
            limit_k=data.get("limit_k", 5),
            top_n=data.get("top_n", 5)
        )

    @classmethod
    def from_json(cls, json_str: str) -> "QueryRequest":
        """Create QueryRequest from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class RerankedResult:
    """Final result after reranking."""
    chunk_id: str
    page_content: str
    original_score: float
    rerank_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page_content": self.page_content,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata
        }


@dataclass
class PipelineResponse:
    """Complete response from the retrieval pipeline."""
    query: str
    retrieved_count: int
    reranked_count: int
    results: list[RerankedResult]
    processing_time_ms: float

    def to_json(self) -> str:
        """Serialize response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Return response as dictionary."""
        return {
            "query": self.query,
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "results": [r.to_dict() for r in self.results],
            "processing_time_ms": self.processing_time_ms
        }


class RetrievalPipeline:
    """
    Orchestrates the complete retrieval pipeline for RAG queries.

    Flow:
    1. Search Qdrant for similar vectors
    2. Rerank results using cross-encoder
    3. Return formatted response
    """

    def __init__(self):
        logfire.info("RetrievalPipeline initialized")

    def process_query(self, request: QueryRequest) -> PipelineResponse:
        """
        Process a validated user query through the complete retrieval pipeline.

        Args:
            request: A validated QueryRequest object.

        Returns:
            PipelineResponse object containing retrieval and reranking results.
        """
        start_time = perf_counter()

        logfire.info(
            "Processing query through retrieval pipeline",
            query=request.query,
            limit_k=request.limit_k,
            top_n=request.top_n,
        )

        try:
            # Step 1: Search Qdrant (includes query embedding internally)
            retrieved_chunks = search_qdrant(
                query=request.query,
                limit_k=request.limit_k
            )

            if not retrieved_chunks:
                logfire.warn("No chunks retrieved from Qdrant", query=request.query)
                return PipelineResponse(
                    query=request.query,
                    retrieved_count=0,
                    reranked_count=0,
                    results=[],
                    processing_time_ms=(perf_counter() - start_time) * 1000
                )

            # Step 2: Rerank documents
            reranked_chunks = rerank_documents(
                query=request.query,
                documents=retrieved_chunks,
                top_n=request.top_n
            )

            final_results = [
                RerankedResult(
                    chunk_id=chunk.chunk_id,
                    page_content=chunk.page_content,
                    original_score=chunk.original_score,
                    rerank_score=chunk.rerank_score,
                    metadata=chunk.metadata,
                )
                for chunk in reranked_chunks
            ]

            processing_time_ms = (perf_counter() - start_time) * 1000

            response = PipelineResponse(
                query=request.query,
                retrieved_count=len(retrieved_chunks),
                reranked_count=len(final_results),
                results=final_results,
                processing_time_ms=processing_time_ms
            )

            logfire.info(
                "Retrieval pipeline completed",
                query=request.query,
                retrieved=len(retrieved_chunks),
                reranked=len(final_results),
                processing_time_ms=processing_time_ms
            )

            return response

        except Exception as exc:
            logfire.error(
                "Retrieval pipeline failed",
                query=request.query,
                error=str(exc)
            )
            raise


# CLI entry point for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retrieval_pipeline.py '<json_query>'")
        print("Example: python retrieval_pipeline.py '{\"query\": \"What is RAG?\", \"limit_k\": 5, \"top_n\": 3}'")
        sys.exit(1)

    json_input = sys.argv[1]

    try:
        pipeline = RetrievalPipeline()
        # Parse boundary input explicitly before passing to process_query
        request = QueryRequest.from_json(json_input)
        response = pipeline.process_query(request)
        print(response.to_json())
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)