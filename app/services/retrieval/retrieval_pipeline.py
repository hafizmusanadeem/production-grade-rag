print("Retrieval_Pipeline.py is just Imported")
"""
Retrieval Pipeline for RAG System.

This module orchestrates the complete retrieval flow:
1. Accepts a user query in JSON format
2. Generates embeddings for the query using the embeddings service
3. Searches Qdrant for relevant chunks
4. Reranks the retrieved results using FlashRank cross-encoder
5. Returns the final reranked results

Usage:
    from app.services.retrieval.retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    result = pipeline.process_query({
        "query": "What is the capital of France?",
        "limit_k": 5,
        "top_n": 3
    })
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional

import logfire

from app.services.retrieval.embeddings import embed_query
from app.services.retrieval.qdrant_service import search_qdrant, RetrievedChunk
from app.services.retrieval.reranker import rerank_documents


@dataclass
class QueryRequest:
    """Input request model for the retrieval pipeline."""
    query: str
    limit_k: int = 5          # Number of chunks to retrieve from Qdrant
    top_n: int = 5            # Number of results after reranking
    collection_name: Optional[str] = None  # Optional override for collection

    @classmethod
    def from_json(cls, json_str: str) -> "QueryRequest":
        """Create QueryRequest from JSON string."""
        data = json.loads(json_str)
        return cls(
            query=data.get("query", ""),
            limit_k=data.get("limit_k", 5),
            top_n=data.get("top_n", 5),
            collection_name=data.get("collection_name")
        )


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline before reranking."""
    chunk_id: str
    page_content: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RerankedResult:
    """Final result after reranking."""
    page_content: str
    original_score: float
    rerank_score: Optional[float] = None
    metadata: dict[str, Any] = None
    chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page_content": self.page_content,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata or {}
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
        return json.dumps({
            "query": self.query,
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "results": [r.to_dict() for r in self.results],
            "processing_time_ms": self.processing_time_ms
        }, indent=2)

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
    1. Embed the user query
    2. Search Qdrant for similar vectors
    3. Rerank results using cross-encoder
    4. Return formatted response
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logfire.info("RetrievalPipeline initialized")

    def process_query(
        self,
        request: QueryRequest | dict | str,
        return_json: bool = False
    ) -> PipelineResponse | str:
        """
        Process a user query through the complete retrieval pipeline.

        Args:
            request: QueryRequest object, dict, or JSON string containing:
                - query: User's natural language query (required)
                - limit_k: Number of chunks to retrieve from Qdrant (default: 5)
                - top_n: Number of results after reranking (default: 5)
                - collection_name: Optional override for Qdrant collection
            return_json: If True, return JSON string instead of PipelineResponse

        Returns:
            PipelineResponse object or JSON string with results
        """
        import time
        start_time = time.time()

        # Parse input
        if isinstance(request, str):
            query_request = QueryRequest.from_json(request)
        elif isinstance(request, dict):
            query_request = QueryRequest(
                query=request.get("query", ""),
                limit_k=request.get("limit_k", 5),
                top_n=request.get("top_n", 5),
                collection_name=request.get("collection_name")
            )
        elif isinstance(request, QueryRequest):
            query_request = request
        else:
            raise ValueError(f"Invalid request type: {type(request)}")

        # Validate query
        if not query_request.query or not query_request.query.strip():
            raise ValueError("Query cannot be empty")

        logfire.info(
            "Processing query through retrieval pipeline",
            query=query_request.query,
            limit_k=query_request.limit_k,
            top_n=query_request.top_n,
        )

        try:
            # Step 1: Search Qdrant (includes query embedding internally)
            retrieved_chunks = search_qdrant(
                query=query_request.query,
                limit_k=query_request.limit_k
            )

            if not retrieved_chunks:
                logfire.warn("No chunks retrieved from Qdrant", query=query_request.query)
                response = PipelineResponse(
                    query=query_request.query,
                    retrieved_count=0,
                    reranked_count=0,
                    results=[],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                return response.to_json() if return_json else response

            # Step 2: Prepare documents for reranking
            documents = [chunk.page_content for chunk in retrieved_chunks]

            # Step 3: Rerank using FlashRank cross-encoder
            reranked_texts = rerank_documents(
                query=query_request.query,
                documents=documents,
                top_n=query_request.top_n
            )

            # Step 4: Map reranked results back to original chunks with metadata
            # Create a mapping from content to original chunk for metadata preservation
            content_to_chunk = {chunk.page_content: chunk for chunk in retrieved_chunks}

            final_results = []
            for rank, reranked_text in enumerate(reranked_texts):
                original_chunk = content_to_chunk.get(reranked_text)
                if original_chunk:
                    result = RerankedResult(
                        chunk_id=original_chunk.chunk_id,
                        page_content=reranked_text,
                        original_score=original_chunk.score,
                        metadata=original_chunk.metadata
                    )
                    final_results.append(result)

            # Handle case where reranker returns fewer results or different texts
            if len(final_results) < len(reranked_texts):
                logfire.warn(
                    "Some reranked texts could not be mapped to original chunks",
                    reranked_count=len(reranked_texts),
                    mapped_count=len(final_results)
                )

            processing_time_ms = (time.time() - start_time) * 1000

            response = PipelineResponse(
                query=query_request.query,
                retrieved_count=len(retrieved_chunks),
                reranked_count=len(final_results),
                results=final_results,
                processing_time_ms=processing_time_ms
            )

            logfire.info(
                "Retrieval pipeline completed",
                query=query_request.query,
                retrieved=len(retrieved_chunks),
                reranked=len(final_results),
                processing_time_ms=processing_time_ms
            )

            return response.to_json() if return_json else response

        except Exception as exc:
            logfire.error(
                "Retrieval pipeline failed",
                query=query_request.query,
                error=str(exc)
            )
            raise

    def process_query_simple(
        self,
        query: str,
        limit_k: int = 5,
        top_n: int = 5
    ) -> list[str]:
        """
        Simplified interface that returns just the reranked text content.

        Args:
            query: User's natural language query
            limit_k: Number of chunks to retrieve from Qdrant
            top_n: Number of results after reranking

        Returns:
            List of reranked document texts
        """
        request = QueryRequest(query=query, limit_k=limit_k, top_n=top_n)
        response = self.process_query(request)
        return [r.page_content for r in response.results]


def create_pipeline() -> RetrievalPipeline:
    """Factory function to create a RetrievalPipeline instance."""
    return RetrievalPipeline()


# Convenience function for direct usage
def retrieve_and_rerank(
    query: str,
    limit_k: int = 5,
    top_n: int = 5,
    as_json: bool = False
) -> PipelineResponse | str:
    """
    Convenience function to run the full retrieval pipeline.

    Args:
        query: User's natural language query
        limit_k: Number of chunks to retrieve from Qdrant
        top_n: Number of results after reranking
        as_json: If True, return JSON string

    Returns:
        PipelineResponse or JSON string
    """
    pipeline = RetrievalPipeline()
    request = QueryRequest(query=query, limit_k=limit_k, top_n=top_n)
    return pipeline.process_query(request, return_json=as_json)


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
        result = pipeline.process_query(json_input, return_json=True)
        print(result)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)