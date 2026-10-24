"""evals/tools/promote_trace_to_golden.py -- turn a reviewed failing trace into
a draft entry in the right golden file (human review required before committing).

Reads LangSmith runs, finds unsampled traces, asks the operator for question/ideal-answer/source,
and writes a draft entry into the corresponding golden file with status='draft'.
Drafts should NOT be committed without human review.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = ROOT / "evals" / "goldens"
RETRIEVER_GOLDEN = GOLDEN_DIR / "retriever.json"
CORRECTNESS_GOLDEN = GOLDEN_DIR / "correctness.json"


def _sample_trace(client):
    runs = client.list_runs(project_name="cx doubt solver", is_root=True, run_type="chain")
    for run in runs:
        fb = client.list_feedback(run_ids=[run.id])
        if fb:
            return run
    return None


def main():
    os.environ.setdefault("LANGSMITH_PROJECT", "cx doubt solver")
    from langsmith import Client
    client = Client()
    run = _sample_trace(client)
    if not run:
        print("No unsampled traces found.")
        raise SystemExit(0)

    outputs = run.outputs or {}
    question = outputs.get("query", "")
    answer = outputs.get("answer", "")
    print(f"Found trace: q={question[:60]}…  answer={answer[:50]}…")
    print("Enter ideal answer (or leave blank to skip):")
    ideal = input().strip()
    if not ideal:
        print("Skipping.")
        raise SystemExit(0)

    print("Enter source file name (e.g. cronjobs.docx):")
    source = input().strip()

    next_id = 0
    path = None
    for name, p in [("retriever", RETRIEVER_GOLDEN), ("correctness", CORRECTNESS_GOLDEN)]:
        full = GOLDEN_DIR / name if name == "retriever" else CORRECTNESS_GOLDEN
        if full.exists() and p is not None:
            path = p
            data = json.loads(p.read_text())
            next_id = len(data) + 1
            break

    entry = {
        "id": f"tr{next_id}",
        "question": question,
        "ideal_answer": ideal,
        "source": source,
        "status": "draft",           # human must review before promoting
        "note": "promoted from live trace; pending human review",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data + [entry], f, indent=2, ensure_ascii=False)

    print(f"Wrote {entry['id']} to {path}")
    print("Review before committing.")


if __name__ == "__main__":
    import os
    main()
