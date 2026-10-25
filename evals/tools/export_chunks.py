"""evals/tools/export_chunks.py -- dump the Qdrant corpus to JSON sorted by file_name."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = ROOT / "evals" / "goldens"
ARTIFACTS_DIR = ROOT / "evals" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = ARTIFACTS_DIR / "chunks_dump.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    from app.config import settings, validate_env_vars
    from app.services.retrieval.qdrant_service import get_qdrant_client
    from evals.helper.qdrant_scan import scroll_all
    validate_env_vars(require_logfire=False, require_llm=False)

    client = get_qdrant_client(timeout=settings.QDRANT_SEARCH_TIMEOUT_SECONDS)
    collection = settings.QDRANT_COLLECTION
    print(f"Reading collection {collection}...")
    docs = []
    for point in scroll_all(client, collection):
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
            "file_name": file_name,
            "chunk_index": payload.get("chunk_index", 0),
            "doc_index": payload.get("doc_index", 0),
            "page": payload.get("page", payload.get("slide")),
            "metadata": metadata,
        })

    docs.sort(key=lambda d: (d["file_name"] or "", d["doc_index"] or 0, d["chunk_index"] or 0))
    if args.max is not None:
        docs = docs[:max(args.max, 0)]

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
