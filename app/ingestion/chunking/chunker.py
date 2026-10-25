from dataclasses import dataclass, field
from typing import Any

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.loaders.base import LoadedDocument


@dataclass
class Chunk:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


@logfire.instrument("Chunk loaded documents", extract_args=False)
def chunk_documents(documents: list[LoadedDocument]) -> list[Chunk]:
    if not documents:
        logfire.warn("No documents provided for chunking")
        return []

    splitter = _build_splitter()
    chunks: list[Chunk] = []

    for doc_index, document in enumerate(documents):
        doc_chunks = splitter.split_text(document.page_content)
        for chunk_index, text in enumerate(doc_chunks):
            metadata = {
                **document.metadata,
                "doc_index": doc_index,
                "chunk_index": chunk_index,
            }
            chunks.append(Chunk(page_content=text, metadata=metadata))

    logfire.info(
        "Chunking complete",
        input_documents=len(documents),
        chunk_count=len(chunks),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return chunks
