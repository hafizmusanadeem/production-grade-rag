"""evals/helper/check_ingestion.py -- verify the Qdrant corpus covers every golden source."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = ROOT / "evals" / "goldens" / "retriever.json"
CONFIG_PATH = ROOT / "app" / "config.py"


def main():
    from app.config import settings, validate_env_vars
    validate_env_vars()
    from app.services.retrieval.qdrant_service import get_qdrant_client

    goldens = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    sources_expected = {g["source"] for g in goldens if g["source"]}

    client = get_qdrant_client(timeout=settings.QDRANT_SEARCH_TIMEOUT_SECONDS)
    collection = settings.QDRANT_COLLECTION
    response = client.get_points(collection_name=collection)

    points = response.points
    by_file = {}
    for p in points:
        payload = dict(p.payload or {})
        fn = payload.get("file_name", "") or payload.get("document_name", "")
        if not fn:
            # fall back to any point with a file_name field
            by_file.setdefault("unknown", []).append(p.id)
            continue
        by_file.setdefault(fn, 0)
        by_file[fn] += 1

    print(f"Golden sources: {len(sources_expected)}")
    for src in sorted(sources_expected):
        found = src in by_file
        pts = by_file.get(src, 0)
        flag = "OK" if found else "MISSING"
        print(f"  {src:<30} points={pts:<5}  [{flag}]")

    missing = sources_expected - set(by_file.keys())
    if missing:
        print(f"\nMissing from corpus: {sorted(missing)}")
        raise SystemExit(1)
    else:
        print("\nAll golden sources present in corpus.")


if __name__ == "__main__":
    main()
