"""
Confirms every `source` cited by the goldens is actually present in Qdrant.

Run:  python -m evals.helper.check_ingestion
Exit code 1 if any golden source is missing.
"""
import evals.helper.eval_env

import json
import sys
from collections import Counter
from pathlib import Path

from app.config import settings
from app.services.retrieval.qdrant_service import get_qdrant_client

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "goldens" / "retriever.json"


def count_points_by_file() -> Counter:
    client = get_qdrant_client(timeout=settings.QDRANT_INGEST_TIMEOUT_SECONDS)
    counts: Counter = Counter()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=["file_name"],
            with_vectors=False,
        )
        for point in points:
            counts[(point.payload or {}).get("file_name", "<missing file_name>")] += 1
        if offset is None:
            break
    return counts


def main() -> int:
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    golden_counts = Counter(g["source"] for g in goldens)
    in_qdrant = count_points_by_file()

    print(f"Collection: {settings.QDRANT_COLLECTION}\n")
    print(f"{'file_name':<28}{'qdrant points':>15}{'golden queries':>17}")
    for name in sorted(set(in_qdrant) | set(golden_counts)):
        print(f"{name:<28}{in_qdrant.get(name, 0):>15}{golden_counts.get(name, 0):>17}")

    missing = sorted(s for s in golden_counts if in_qdrant.get(s, 0) == 0)
    if missing:
        print(f"\n❌ Golden sources NOT in Qdrant: {', '.join(missing)}")
        print("   Ingest only these (see note below), then re-run this check.")
        return 1

    print("\n✅ All golden sources are present in Qdrant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
