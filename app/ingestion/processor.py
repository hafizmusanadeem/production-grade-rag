import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.ingestion.chunking.chunker import chunk_documents
from app.ingestion.loaders import load_document
from app.services.retrieval.embeddings.embeddings import EmbeddedChunk, embed_chunks


@dataclass
class ProcessResult:
    source: str
    chunk_count: int
    local_path: str
    qdrant_collection: str
    qdrant_points_upserted: int


class IngestionProcessor:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or "DATA")
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._qdrant: QdrantClient | None = None

    @logfire.instrument("Process file {file_path=}", extract_args=("file_path",))
    def process(self, file_path: str | Path) -> ProcessResult:
        path = Path(file_path)
        documents = load_document(path)
        chunks = chunk_documents(documents)
        embedded_chunks = embed_chunks(chunks)

        local_path = self._store_locally(path, embedded_chunks)
        qdrant_points = self._store_in_qdrant(embedded_chunks)

        result = ProcessResult(
            source=str(path.resolve()),
            chunk_count=len(embedded_chunks),
            local_path=str(local_path),
            qdrant_collection=settings.QDRANT_COLLECTION or "",
            qdrant_points_upserted=qdrant_points,
        )
        logfire.info("Ingestion complete", **asdict(result))
        return result

    def _store_locally(self, source_path: Path, chunks: list[EmbeddedChunk]) -> Path:
        output_path = self.embeddings_dir / f"{source_path.stem}.embeddings.json"
        payload = {
            "source": str(source_path.resolve()),
            "file_name": source_path.name,
            "created_at": datetime.now(UTC).isoformat(),
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page_content": chunk.page_content,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logfire.info(
            "Stored embeddings locally",
            output_path=str(output_path),
            chunk_count=len(chunks),
        )
        return output_path

    def _store_in_qdrant(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0

        client = self._get_qdrant_client()
        collection_name = settings.QDRANT_COLLECTION
        if not collection_name:
            raise ValueError("QDRANT_COLLECTION is required to store embeddings")

        vector_size = len(chunks[0].embedding)
        self._ensure_collection(client, collection_name, vector_size)

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

        client.upsert(collection_name=collection_name, points=points)
        logfire.info(
            "Stored embeddings in Qdrant",
            collection=collection_name,
            point_count=len(points),
            vector_size=vector_size,
        )
        return len(points)

    def _get_qdrant_client(self) -> QdrantClient:
        if self._qdrant is not None:
            return self._qdrant

        if not settings.QDRANT_CLUSTER_ENDPOINT:
            raise ValueError("QDRANT_CLUSTER_ENDPOINT is required to store embeddings")
        if not settings.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY is required to store embeddings")

        self._qdrant = QdrantClient(
            url=settings.QDRANT_CLUSTER_ENDPOINT,
            api_key=settings.QDRANT_API_KEY,
        )
        return self._qdrant

    def _ensure_collection(
        self,
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


def process_file(file_path: str | Path, data_dir: str | Path | None = None) -> ProcessResult:
    return IngestionProcessor(data_dir=data_dir).process(file_path)
