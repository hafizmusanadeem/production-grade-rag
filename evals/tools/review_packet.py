"""Build the human golden-review packet (Markdown + CSV per golden file).

RED first, then YELLOW, then GREEN, so a reviewer sees the risky items first.
Every row leaves `decision` and `notes` blank for the human.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "evals" / "artifacts" / "chunks_dump.json"
GOLDEN_DIR = ROOT / "evals" / "goldens"
REVIEW_DIR = ROOT / "evals" / "artifacts" / "review"
SWEEP = ROOT / "evals" / "artifacts" / "threshold_sweep.json"

# Files whose items answer a question from a specific ingested document.
SOURCED_FILES = ("retriever.json", "correctness.json", "faithfulness.json")
# Files whose items assert behaviour rather than a fact.
BEHAVIOUR_FILES = ("scope_goldens.json", "leakage_goldens.json", "toxicity_goldens.json",
                   "routing.json", "unanswerable.json", "multiturn.json", "ops_questions.json")

ANSWER_KEY = {"retriever.json": "ideal_answer", "correctness.json": "ideal_answer",
              "faithfulness.json": "ideal_context"}
QUESTION_KEY = {"retriever.json": "query", "correctness.json": "question",
                "faithfulness.json": "query"}


def tokens(text):
    import re
    return {t for t in re.findall(r"[a-zA-Z0-9_./-]+", str(text).lower())
            if len(t) >= 3 or any(ch.isdigit() for ch in t) or "_" in t or "." in t}


def key_tokens(text):
    import re
    return set(re.findall(r'"[^"]+"|\b[A-Za-z][A-Za-z0-9_.-]{2,}\b|\b\d+(?:\.\d+)*\b', str(text)))


def load_chunks():
    if not DUMP.exists():
        raise SystemExit(f"Missing {DUMP}; run python -m evals.tools.export_chunks first.")
    chunks = json.loads(DUMP.read_text(encoding="utf-8"))
    by_file = {}
    for chunk in chunks:
        by_file.setdefault(chunk.get("file_name") or "unknown", []).append(chunk)
    return by_file


def answer_text(name, golden):
    value = golden.get(ANSWER_KEY[name], "")
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def source_text(by_file, source):
    return " ".join(c.get("page_content", "") for c in by_file.get(source, [])).lower()


def best_chunk(by_file, source, answer):
    """Highest-token-overlap chunk of `source` against `answer`."""
    ideal = tokens(answer)
    best = (0.0, None)
    for chunk in by_file.get(source, []):
        chunk_tokens = tokens(chunk.get("page_content", ""))
        score = len(ideal & chunk_tokens) / len(ideal) if ideal else 0.0
        if score > best[0]:
            best = (score, chunk)
    return best


def ungrounded_tokens(by_file, source, answer):
    src = source_text(by_file, source)
    return sorted(t for t in key_tokens(answer) if t.strip('"').lower() not in src)


def traffic_light(overlap, ungrounded_rate):
    """GREEN: overlap >= 0.6 and nothing ungrounded. YELLOW: otherwise.
    RED: overlap < 0.3 or more than 20% ungrounded tokens."""
    if overlap < 0.3 or ungrounded_rate > 0.2:
        return "RED"
    if overlap >= 0.6 and ungrounded_rate == 0:
        return "GREEN"
    return "YELLOW"


def sort_by_light(rows):
    rank = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    return sorted(rows, key=lambda r: (rank.get(r["light"], 3), r["id"]))


def review_row(name, golden, by_file):
    answer = answer_text(name, golden)
    overlap, chunk = best_chunk(by_file, golden.get("source", ""), answer)
    ungrounded = ungrounded_tokens(by_file, golden.get("source", ""), answer)
    n_keys = len(key_tokens(answer))
    rate = (len(ungrounded) / n_keys) if n_keys else 0.0
    return {
        "id": golden.get("id", ""),
        "tags": ";".join(golden.get("tags", []) or []),
        "question": golden.get(QUESTION_KEY[name], "") or golden.get("query", ""),
        "source": golden.get("source", ""),
        "overlap": round(overlap, 3),
        "ungrounded_count": len(ungrounded),
        "ungrounded_rate": round(rate, 4),
        "best_chunk_id": (chunk or {}).get("chunk_id", ""),
        "best_chunk_text": (chunk or {}).get("page_content", ""),
        "ungrounded_tokens": "; ".join(ungrounded),
        "light": traffic_light(overlap, rate),
        "ideal_answer": answer,
        "decision": "",
        "notes": "",
    }


BEHAVIOUR_COLUMNS = {
    "scope_goldens.json": ("input", "expected_action", "success_criteria"),
    "leakage_goldens.json": ("input", "expected_action", "success_criteria"),
    "toxicity_goldens.json": ("input", "expected_action", ""),
    "routing.json": ("message", "expected_intent", ""),
    "unanswerable.json": ("question", "expected", ""),
    "multiturn.json": ("expected_behavior", "expected_intent", ""),
    "ops_questions.json": ("tier", "derived_from", ""),
}


def behaviour_row(name, golden, sweeps):
    columns = BEHAVIOUR_COLUMNS.get(name, ("input", "expected_action", "success_criteria"))
    first, second, third = columns
    row = {
        "id": golden.get("id", ""),
        "input": str(golden.get(first, ""))[:300],
        "expected": str(golden.get(second, "")),
        "success_criteria": str(golden.get(third, "")),
        "tags": ";".join(golden.get("tags", []) or []),
        "decision": "",
        "notes": "",
    }
    if name == "unanswerable.json":
        gid = golden.get("id", "")
        row["absence_proof"] = (sweeps.get(gid, {}).get("best_rerank_score") if sweeps
                                else "not computed")
    return row


def load_sweeps():
    if not SWEEP.exists():
        return {}
    try:
        data = json.loads(SWEEP.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    out = {}
    for entry in data if isinstance(data, list) else data.get("items", []):
        gid = entry.get("id") or entry.get("golden_id")
        if gid:
            out[gid] = entry
    return out


def order(rows, name, by_file):
    if name in SOURCED_FILES:
        rows = [review_row(name, g, by_file) for g in rows]
        return sort_by_light(rows)
    sweeps = load_sweeps()
    return [behaviour_row(name, g, sweeps) for g in rows]


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, name, rows, columns, light=None):
    lines = [f"# Golden review packet: {name}", ""]
    if light:
        counts = {c: light.count(c) for c in ("RED", "YELLOW", "GREEN")}
        lines.append("Traffic lights: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        lines.append("")
    for row in rows:
        lines.append(f"## {row.get('id', '')}  ({row.get('light', 'BEHAVIOUR')})")
        lines.append("")
        for col in columns:
            value = row.get(col, "")
            if col in ("best_chunk_text", "ideal_answer"):
                lines.append(f"- **{col}**:")
                lines.append("")
                lines.append("```")
                lines.append(str(value))
                lines.append("```")
            else:
                lines.append(f"- **{col}**: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


SOURCED_COLUMNS = ["id", "light", "tags", "question", "source", "overlap",
                   "ungrounded_rate", "best_chunk_id", "ungrounded_tokens",
                   "decision", "notes"]
SOURCED_MD = SOURCED_COLUMNS + ["best_chunk_text", "ideal_answer"]
BEHAVIOUR_MD = ["id", "tags", "input", "expected", "success_criteria",
                "absence_proof", "decision", "notes"]


def main():
    ap = argparse.ArgumentParser(description="Build the human golden-review packet.")
    ap.add_argument("--only", default=None, help="comma-separated golden file names")
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None

    by_file = load_chunks()
    names = only or [p.name for p in sorted(GOLDEN_DIR.glob("*.json"))]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        path = GOLDEN_DIR / name
        if not path.exists():
            print(f"skipping {name}: not found")
            continue
        goldens = json.loads(path.read_text(encoding="utf-8"))
        if name in SOURCED_FILES:
            rows = order(goldens, name, by_file)
            light = [r["light"] for r in rows]
            write_md(REVIEW_DIR / f"{name}.md", name, rows, SOURCED_MD, light)
            write_csv(REVIEW_DIR / f"{name}.csv", rows, SOURCED_COLUMNS)
            summary[name] = {c: light.count(c) for c in ("RED", "YELLOW", "GREEN")}
        else:
            rows = order(goldens, name, by_file)
            cols = [c for c in BEHAVIOUR_MD if any(r.get(c) for r in rows)] or BEHAVIOUR_MD
            write_md(REVIEW_DIR / f"{name}.md", name, rows, cols)
            write_csv(REVIEW_DIR / f"{name}.csv", rows,
                      [c for c in BEHAVIOUR_MD if c != "best_chunk_text"])
            summary[name] = {"items": len(rows)}

    print("\nReview packet summary")
    print("-" * 48)
    for name, counts in summary.items():
        print(f"{name:<28} {counts}")
    print("-" * 48)
    print(f"Wrote {len(summary)} packet(s) to {REVIEW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
