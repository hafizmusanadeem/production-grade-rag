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
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import logfire

from app.config import RERANK_TOP_N, RETRIEVAL_LIMIT_K
from app.exceptions import RAGError, RerankError, RetrievalError
from app.services.retrieval.qdrant_service import search_qdrant
from app.services.retrieval.reranker import rerank_documents


@dataclass
class QueryRequest:
    """Input request model for the retrieval pipeline."""
    query: str
    limit_k: int = RETRIEVAL_LIMIT_K          # Number of chunks to retrieve from Qdrant
    top_n: int = RERANK_TOP_N            # Number of results after reranking
    use_reranker: bool = True         # Whether to run semantic reranking
    min_rerank_score: float | None = None

    def __post_init__(self):
        """Validate fields upon instantiation (Fail Fast)."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty or whitespace")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryRequest":
        """Create QueryRequest from a dictionary."""
        return cls(
            query=data.get("query", ""),
            limit_k=data.get("limit_k", RETRIEVAL_LIMIT_K),
            top_n=data.get("top_n", RERANK_TOP_N),
            use_reranker=data.get("use_reranker", True),
            min_rerank_score=data.get("min_rerank_score")
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
    timings_ms: dict = field(default_factory=dict)
    top_rerank_score: float | None = None

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
            "processing_time_ms": self.processing_time_ms,
            "timings_ms": self.timings_ms,
            "top_rerank_score": self.top_rerank_score,
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
            search_started = perf_counter()
            try:
                retrieved_chunks = search_qdrant(query=request.query, limit_k=request.limit_k)
            except RAGError:
                raise
            except Exception as exc:
                raise RetrievalError(f"Vector search failed: {exc}") from exc
            embed_qdrant_ms = (perf_counter() - search_started) * 1000
            if not retrieved_chunks:
                logfire.warn("No chunks retrieved from Qdrant", query=request.query)
                return PipelineResponse(
                    query=request.query,
                    retrieved_count=0,
                    reranked_count=0,
                    results=[],
                    processing_time_ms=(perf_counter() - start_time) * 1000,
                    timings_ms={"embed_qdrant": embed_qdrant_ms},
                )

            rerank_ms = 0.0
            top_rerank_score = None
            if request.use_reranker:
                rerank_started = perf_counter()
                try:
                    reranked_chunks = rerank_documents(query=request.query, documents=retrieved_chunks, top_n=request.top_n)
                except RAGError:
                    raise
                except Exception as exc:
                    raise RerankError(f"Reranking failed: {exc}") from exc
                rerank_ms = (perf_counter() - rerank_started) * 1000
                top_rerank_score = max((float(c.rerank_score) for c in reranked_chunks), default=None)
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
                if request.min_rerank_score is not None:
                    final_results = [r for r in final_results if r.rerank_score >= request.min_rerank_score]
            else:
                final_results = [
                    RerankedResult(
                        chunk_id=chunk.chunk_id,
                        page_content=chunk.page_content,
                        original_score=chunk.score,
                        rerank_score=chunk.score,
                        metadata=chunk.metadata,
                    )
                    for chunk in retrieved_chunks[: request.top_n]
                ]
            processing_time_ms = (perf_counter() - start_time) * 1000

            response = PipelineResponse(
                query=request.query,
                retrieved_count=len(retrieved_chunks),
                reranked_count=len(final_results),
                results=final_results,
                processing_time_ms=processing_time_ms,
                timings_ms={"embed_qdrant": embed_qdrant_ms, "rerank": rerank_ms},
                top_rerank_score=top_rerank_score,
            )

            logfire.info(
                "Retrieval pipeline completed",
                query=request.query,
                use_reranker=request.use_reranker,
                retrieved=len(retrieved_chunks),
                reranked=len(final_results),
                processing_time_ms=processing_time_ms,
                timings_ms={"embed_qdrant": embed_qdrant_ms, "rerank": rerank_ms},
                top_rerank_score=top_rerank_score,
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