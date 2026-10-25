"""Draft-status guard: no baseline may be built on unreviewed goldens."""
import argparse
import glob
import json
import os

from evals.config import GOLDEN_DIR

DRAFT_MARK = "draft"


def counts_for_file(path):
    """Return (total, drafts) for one golden file; non-JSON files give (0, 0)."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        return 0, 0
    drafts = sum(1 for item in items
                 if isinstance(item, dict) and str(item.get("status", "")).lower() == DRAFT_MARK)
    return len(items), drafts


def scan(golden_dir=GOLDEN_DIR):
    rows = []
    for path in sorted(glob.glob(os.path.join(str(golden_dir), "*.json"))):
        try:
            total, drafts = counts_for_file(path)
        except (ValueError, OSError):
            totals = 0
            rows.append({"file": os.path.basename(path), "total": 0, "draft": -1, "error": "unreadable"})
            continue
        rows.append({"file": os.path.basename(path), "total": total, "draft": drafts, "error": None})
    return rows


def render(rows):
    lines = [f"{'golden file':<32} {'total':>6} {'draft':>6}", "-" * 48]
    for row in rows:
        if row["error"]:
            lines.append(f"{row['file']:<32} {row['total']:>6} {'ERROR':>6}  {row['error']}")
        else:
            lines.append(f"{row['file']:<32} {row['total']:>6} {row['draft']:>6}")
    total_draft = sum(row["draft"] for row in rows if row["draft"] >= 0)
    lines.append("-" * 48)
    lines.append(f"files={len(rows)}  draft items={total_draft}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Report unreviewed (draft) golden items.")
    ap.add_argument("--fail-on-draft", action="store_true",
                    help="exit 1 when any item still has status=draft")
    args = ap.parse_args()
    rows = scan()
    print(render(rows))
    total_draft = sum(row["draft"] for row in rows if row["draft"] >= 0)
    if args.fail_on_draft and total_draft:
        print(f"BLOCKED: {total_draft} golden item(s) still marked draft; a human must review them.")
        return 1
    if args.fail_on_draft:
        print("OK: every golden item is human-reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
