# processor.py

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import logfire

from app.config import settings
from app.ingestion.chunking.chunker import chunk_documents
from app.ingestion.loaders import load_document
from app.services.retrieval.embeddings import EmbeddedChunk, embed_chunks
from app.services.retrieval.qdrant_service import store_embeddings


@dataclass(frozen=True)
class ProcessResult:
    """Result of processing a single document."""

    source: str
    chunk_count: int
    local_path: str
    qdrant_collection: str
    qdrant_points_upserted: int


class IngestionProcessor:
    """
    Orchestrates the document ingestion pipeline.

    Pipeline:

        source file
            ↓
        document loading
            ↓
        chunking
            ↓
        embedding
            ↓
        Qdrant persistence
            ↓
        local embedding checkpoint

    The local embedding file is intentionally written only after the
    Qdrant upsert succeeds. This prevents an incomplete Qdrant ingestion
    from being incorrectly treated as a completed checkpoint by the
    batch processor.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or "DATA")
        self.embeddings_dir = self.data_dir / "embeddings"

        self.embeddings_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logfire.info(
            "Ingestion processor initialized",
            data_directory=str(self.data_dir.resolve()),
            embeddings_directory=str(self.embeddings_dir.resolve()),
        )

    @logfire.instrument(
        "Process file {file_path=}",
        extract_args=("file_path",),
    )
    def process(
        self,
        file_path: str | Path,
    ) -> ProcessResult:
        """
        Process a single document through the complete ingestion pipeline.

        The operation is considered successful only when:
        1. The document loads successfully.
        2. Chunks are generated.
        3. Embeddings are generated.
        4. All embeddings are successfully upserted to Qdrant.
        5. The local embedding checkpoint is successfully written.

        If Qdrant persistence fails, the local checkpoint is NOT written.
        This is critical because the batch processor uses the existence of
        local embedding files as a processing checkpoint.
        """

        path = self._validate_source_path(file_path)

        started_at = datetime.now(UTC)

        logfire.info(
            "Starting document ingestion",
            source=str(path),
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
        )

        # ------------------------------------------------------------------
        # STEP 1: LOAD
        # ------------------------------------------------------------------

        documents = load_document(path)

        if not documents:
            raise ValueError(
                f"Document loader returned no documents for: {path}"
            )

        logfire.info(
            "Document loaded",
            source=str(path),
            document_count=len(documents),
        )

        # ------------------------------------------------------------------
        # STEP 2: CHUNK
        # ------------------------------------------------------------------

        chunks = chunk_documents(documents)

        if not chunks:
            raise ValueError(
                f"No chunks were generated for: {path}"
            )

        logfire.info(
            "Document chunked",
            source=str(path),
            document_count=len(documents),
            chunk_count=len(chunks),
        )

        # ------------------------------------------------------------------
        # STEP 3: EMBED
        # ------------------------------------------------------------------

        embedded_chunks = embed_chunks(chunks)

        if not embedded_chunks:
            raise ValueError(
                f"No embeddings were generated for: {path}"
            )

        logfire.info(
            "Document embeddings generated",
            source=str(path),
            chunk_count=len(embedded_chunks),
            vector_dimension=len(embedded_chunks[0].embedding),
            embedding_model=settings.GEMINI_EMBEDDING_MODEL,
        )

        # ------------------------------------------------------------------
        # STEP 4: QDRANT
        # ------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # This happens BEFORE _store_locally().
        #
        # The batch processor treats the local embedding file as a
        # checkpoint. Therefore, writing it before Qdrant succeeds creates
        # a false checkpoint if Qdrant fails or times out.
        #
        # Your previous failure was exactly this:
        #
        #     local embeddings → written successfully
        #     Qdrant           → timeout
        #
        # Result:
        #
        #     batch processor sees the local file
        #     → assumes document is complete
        #     → skips it
        #
        # ------------------------------------------------------------------

        logfire.info(
            "Persisting embeddings to Qdrant",
            source=str(path),
            chunk_count=len(embedded_chunks),
            collection=settings.QDRANT_COLLECTION or "unknown",
        )

        try:
            qdrant_points = store_embeddings(embedded_chunks)
        except Exception:
            logfire.exception(
                "Qdrant persistence failed",
                source=str(path),
                chunk_count=len(embedded_chunks),
                collection=settings.QDRANT_COLLECTION or "unknown",
            )
            raise

        if qdrant_points != len(embedded_chunks):
            raise RuntimeError(
                "Qdrant persistence returned an unexpected point count: "
                f"expected {len(embedded_chunks)}, "
                f"received {qdrant_points}"
            )

        logfire.info(
            "Qdrant persistence completed",
            source=str(path),
            points_upserted=qdrant_points,
            collection=settings.QDRANT_COLLECTION or "unknown",
        )

        # ------------------------------------------------------------------
        # STEP 5: LOCAL CHECKPOINT
        # ------------------------------------------------------------------
        #
        # Only write this checkpoint after Qdrant has successfully accepted
        # all points.
        #
        # This ordering is essential for batch resume correctness.
        # ------------------------------------------------------------------

        local_path = self._store_locally(
            source_path=path,
            chunks=embedded_chunks,
        )

        # ------------------------------------------------------------------
        # FINAL RESULT
        # ------------------------------------------------------------------

        processing_time = (
            datetime.now(UTC) - started_at
        ).total_seconds()

        result = ProcessResult(
            source=str(path),
            chunk_count=len(embedded_chunks),
            local_path=str(local_path),
            qdrant_collection=settings.QDRANT_COLLECTION or "",
            qdrant_points_upserted=qdrant_points,
        )

        logfire.info(
            "Ingestion complete",
            **asdict(result),
            processing_time_sec=processing_time,
        )

        return result

    def _validate_source_path(
        self,
        file_path: str | Path,
    ) -> Path:
        """
        Validate and normalize the source file path.

        Validation happens before invoking the loader so that filesystem
        errors are surfaced consistently by the ingestion processor.
        """

        if file_path is None:
            raise ValueError("file_path is required")

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Source file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Source path is not a file: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"Source file is empty: {path}"
            )

        logfire.info(
            "Source file validated",
            source=str(path),
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
        )

        return path

    def _store_locally(
        self,
        source_path: Path,
        chunks: list[EmbeddedChunk],
    ) -> Path:
        """
        Store embeddings locally as an atomic JSON checkpoint.

        The write is performed into a temporary file in the same directory
        and then atomically replaced into the final path. This prevents a
        process interruption from leaving a partially-written JSON file
        that could later be mistaken for a valid checkpoint.
        """

        output_path = (
            self.embeddings_dir
            / f"{source_path.stem}.embeddings.json"
        )

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

        serialized_payload = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.embeddings_dir,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                temporary_file.write(serialized_payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            temporary_path.replace(output_path)

        except Exception:
            logfire.exception(
                "Failed to store local embedding checkpoint",
                output_path=str(output_path),
                source=str(source_path),
                chunk_count=len(chunks),
            )

            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logfire.warning(
                        "Failed to clean temporary embedding file",
                        temporary_path=str(temporary_path),
                    )

            raise

        logfire.info(
            "Stored embeddings locally",
            output_path=str(output_path),
            source=str(source_path),
            chunk_count=len(chunks),
            file_size_bytes=output_path.stat().st_size,
        )

        return output_path


def process_file(
    file_path: str | Path,
    data_dir: str | Path | None = None,
) -> ProcessResult:
    """
    Convenience function for processing a single document.
    """

    return IngestionProcessor(
        data_dir=data_dir,
    ).process(file_path)