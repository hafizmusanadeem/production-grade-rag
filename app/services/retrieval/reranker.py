from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

import logfire
from app.services.retrieval.qdrant_service import RetrievedChunk

if TYPE_CHECKING:
    from flashrank import Ranker


ranker = None


@dataclass
class RerankedChunk:
    chunk_id: str
    page_content: str
    original_score: float
    rerank_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def get_ranker() -> Ranker:
    """Initialize FlashRank lazily and reuse the loaded model."""
    global ranker

    if ranker is None:
        from flashrank import Ranker

        logfire.info("Initializing FlashRank model")

        try:
            ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            ranker = Ranker()

    return ranker


def rerank_documents(
    query: str,
    documents: list[RetrievedChunk],
    top_n: int = 5,
) -> list[RerankedChunk]:
    """Rerank retrieved chunks using FlashRank."""
    if not documents:
        return []

    start_time = perf_counter()

    logfire.info(
        "Sending documents to FlashRank",
        document_count=len(documents),
        top_n=top_n,
    )

    try:
        from flashrank import RerankRequest

        ranker_instance = get_ranker()

        passages = [
            {
                "id": index,
                "text": document.page_content,
            }
            for index, document in enumerate(documents)
        ]

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        results = ranker_instance.rerank(request)

        reranked_documents = []

        for result in results[:top_n]:
            original_document = documents[result["id"]]

            reranked_documents.append(
                RerankedChunk(
                    chunk_id=original_document.chunk_id,
                    page_content=original_document.page_content,
                    original_score=original_document.score,
                    rerank_score=result["score"],
                    metadata=original_document.metadata,
                )
            )

        duration = perf_counter() - start_time

        logfire.info(
            "FlashRank reranking completed",
            document_count=len(documents),
            result_count=len(reranked_documents),
            duration_seconds=duration,
            top_score=(
                results[0]["score"]
                if results
                else None
            ),
        )

        return reranked_documents

    except Exception:
        logfire.exception("FlashRank reranking failed")
        raise