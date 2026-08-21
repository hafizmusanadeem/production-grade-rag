import json
from pathlib import Path

from qdrant_client.http import models as qmodels

from app.config import settings
from app.services.retrieval.qdrant_service import (
    ensure_collection,
    get_qdrant_client,
)


EMBEDDINGS_FILE = Path(
    "DATA/embeddings/architecture.embeddings.json"
)

BATCH_SIZE = 5


def main() -> None:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    payload = json.loads(
        EMBEDDINGS_FILE.read_text(encoding="utf-8")
    )

    chunks = payload["chunks"]

    if not chunks:
        raise ValueError(
            "No chunks found in embeddings file"
        )

    collection_name = settings.QDRANT_COLLECTION

    if not collection_name:
        raise ValueError(
            "QDRANT_COLLECTION is required"
        )

    client = get_qdrant_client()

    vector_size = len(chunks[0]["embedding"])

    ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    points = [
        qmodels.PointStruct(
            id=chunk["chunk_id"],
            vector=chunk["embedding"],
            payload={
                "page_content": chunk["page_content"],
                **chunk["metadata"],
            },
        )
        for chunk in chunks
    ]

    total = len(points)

    print(
        f"Preparing to restore {total} points "
        f"for {payload['file_name']}"
    )

    print(f"Batch size: {BATCH_SIZE}")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch = points[start:end]

        batch_number = (start // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"\nUploading batch "
            f"{batch_number}/{total_batches} "
            f"({start + 1}-{end} of {total})..."
        )

        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

        print(
            f"Batch {batch_number}/{total_batches} "
            f"uploaded successfully."
        )

    final_count = client.get_collection(
        collection_name
    ).points_count

    print(
        f"\nRecovery complete."
    )

    print(
        f"Qdrant collection now contains "
        f"{final_count} points."
    )


if __name__ == "__main__":
    main()