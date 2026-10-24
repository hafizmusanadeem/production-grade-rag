"""evals/tools/export_chunks.py -- dump the Qdrant corpus to JSON sorted by file_name."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = ROOT / "evals" / "goldens"
ARTIFACTS_DIR = ROOT / "evals" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = ARTIFACTS_DIR / "chunks_dump.json"


def main():
    from app.config import settings, validate_env_vars
    from app.services.retrieval.qdrant_service import get_qdrant_client
    validate_env_vars()

    client = get_qdrant_client(timeout=settings.QDRANT_SEARCH_TIMEOUT_SECONDS)
    collection = settings.QDRANT_COLLECTION
    print(f"Reading collection {collection}...")
    response = client.get_points(collection_name=collection)

    docs = []
    for point in response.points:
        payload = dict(point.payload or {})
        page_content = payload.pop("page_content", "")
        if not page_content:
            continue
        chunk_id = str(point.id)
        file_name = payload.get("file_name", "")
        metadata = payload
        docs.append({
            "chunk_id": chunk_id,
            "page_content": page_content,
            "score": float(point.score),
            "original_score": float(payload.get("original_score", 0)),
            "rerank_score": float(payload.get("rerank_score", 0)),
            "metadata": metadata,
            "file_name": file_name,
        })

    # Sort by file_name, then chunk_id for determinism
    docs.sort(key=lambda d: (d["file_name"] or "", d["chunk_id"]))

    # Cap at 5000 to keep the dump manageable
    docs = docs[:5000]

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    by_file = {}
    for d in docs:
        fn = d["file_name"] or "unknown"
        by_file.setdefault(fn, []).append(d)

    print(f"Wrote {len(docs)} chunks to {OUTPUT}")
    print(f"Distinct files: {len(by_file)}")
    for fn, chunks in sorted(by_file.items()):
        print(f"  {fn}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
