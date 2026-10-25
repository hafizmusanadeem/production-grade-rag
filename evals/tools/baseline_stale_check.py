"""Re-baseline guard: is the repo still comparable to the committed baseline?

Warning mode by default (exit 0) so CI can annotate without blocking;
--strict exits 1. Corpus fingerprint is deliberately NOT checked here --
it needs a live Qdrant connection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from evals.config import BASELINE_PATH, GOLDEN_DIR

SETTINGS_KEYS = ("chunk_size", "chunk_overlap", "embedding_model",
                 "retrieval_limit_k", "rerank_top_n", "rerank_min_score")


def current_settings():
    from app.config import RERANK_TOP_N, RETRIEVAL_LIMIT_K, settings
    return {"chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            "retrieval_limit_k": RETRIEVAL_LIMIT_K,
            "rerank_top_n": RERANK_TOP_N,
            "rerank_min_score": settings.RERANK_MIN_SCORE}


def current_golden_hashes(golden_dir=GOLDEN_DIR):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(golden_dir).glob("*.json"))}


def current_prompt_hash():
    try:
        from app.agents.prompts import prompt_fingerprint
        return prompt_fingerprint()
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}>"


def drift(baseline_metadata):
    """Return a list of (field, baseline, current) tuples that differ."""
    rows = []

    bm = baseline_metadata or {}

    pair = ("prompt_hash", bm.get("prompt_hash"), current_prompt_hash())
    if pair[1] != pair[2]:
        rows.append(pair)

    bs = bm.get("settings", {}) or {}
    for key in SETTINGS_KEYS:
        now = current_settings()
        if bs.get(key) != now[key]:
            rows.append((f"settings.{key}", bs.get(key), now[key]))

    bh = bm.get("golden_hashes", {}) or {}
    nh = current_golden_hashes()
    for name in sorted(set(bh) | set(nh)):
        if bh.get(name) != nh.get(name):
            rows.append((f"golden_hashes.{name}",
                         "present" if bh.get(name) else "absent",
                         "present" if nh.get(name) else "absent"))

    bj = bm.get("judge_model", "<unset>")
    nj = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b:free")
    if bj != nj:
        rows.append(("judge_model", bj, nj))

    return rows


def render(rows, baseline_path):
    lines = [f"Baseline staleness check against {baseline_path}", "-" * 72]
    if not rows:
        lines.append("No drift: prompts, settings, goldens and judge model all match the baseline.")
    else:
        lines.append(f"{'field':<38} {'baseline':>16} {'current':>16}")
        lines.append("-" * 72)
        for field, before, after in rows:
            lines.append(f"{field:<38} {str(before)[:16]:>16} {str(after)[:16]:>16}")
        lines.append("-" * 72)
        lines.append(f"{len(rows)} drifted field(s). A re-baseline is needed before comparing.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Check whether a baseline is stale.")
    ap.add_argument("--baseline", default=str(BASELINE_PATH))
    ap.add_argument("--strict", action="store_true", help="exit 1 when drift is found")
    args = ap.parse_args()
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    rows = drift(baseline.get("metadata", {}))
    print(render(rows, args.baseline))
    if args.strict and rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
