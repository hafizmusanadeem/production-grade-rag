#processor.py
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import logfire

from app.config import settings
from app.ingestion.chunking.chunker import chunk_documents
from app.ingestion.loaders import load_document
from app.services.retrieval.qdrant_service import store_embeddings
from app.services.retrieval.embeddings import EmbeddedChunk, embed_chunks


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
        self.embeddings_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @logfire.instrument(
        "Process file {file_path=}",
        extract_args=("file_path",),
    )
    def process(
        self,
        file_path: str | Path,
    ) -> ProcessResult:

        path = Path(file_path)

        documents = load_document(path)

        chunks = chunk_documents(documents)

        embedded_chunks = embed_chunks(chunks)

        local_path = self._store_locally(
            path,
            embedded_chunks,
        )

        qdrant_points = store_embeddings(
            embedded_chunks
        )

        result = ProcessResult(
            source=str(path.resolve()),
            chunk_count=len(embedded_chunks),
            local_path=str(local_path),
            qdrant_collection=settings.QDRANT_COLLECTION or "",
            qdrant_points_upserted=qdrant_points,
        )

        logfire.info(
            "Ingestion complete",
            **asdict(result),
        )

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

   
def process_file(file_path: str | Path, data_dir: str | Path | None = None) -> ProcessResult:
    return IngestionProcessor(data_dir=data_dir).process(file_path)
