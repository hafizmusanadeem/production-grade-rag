"""evals/tools/pick_golden_context.py -- hand-pick ideal context for faithfulness goldens.

Reads the chunks_dump.json (from export_chunks.py) and the retriever goldens,
extracts one good chunk per golden source, writes evals/goldens/faithfulness.json.
Run once to generate; the output is committed. Re-runnable if you change the dump.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP_PATH = ROOT / "evals" / "artifacts" / "chunks_dump.json"
RETRIEVER_PATH = ROOT / "evals" / "goldens" / "retriever.json"
OUT_PATH = ROOT / "evals" / "goldens" / "faithfulness.json"


def main():
    if not DUMP_PATH.exists():
        print(f"chunks_dump.json missing at {DUMP_PATH}. Run export_chunks first.")
        raise SystemExit(1)

    chunks = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
    by_file = {}
    for c in chunks:
        fn = c["file_name"] or "unknown"
        by_file.setdefault(fn, []).append(c)

    goldens = json.load(open(RETRIEVER_PATH, encoding="utf-8"))
    out = []
    for g in goldens[:30]:
        # Pick the highest rerank_score chunk from this source
        src = g["source"]
        cands = by_file.get(src, [])
        if not cands:
            # Try common extensions
            ext_map = {"cronjobs.docx": "cronjobs.docx", "architecture.pptx": "architecture.pptx",
                       "monitor_job.docx": "monitor_job.docx", "parallel_work_queue.txt": "parallel_work_queue.txt",
                       "pods_autoscale.html": "pods_autoscale.html"}
            src2 = ext_map.get(g["source"])
            if src2:
                cands = by_file.get(src2, [])
        if not cands:
            out.append({"id": g["id"], "query": g["query"], "ideal_context": "",
                        "source": src})
            continue
        best = max(cands, key=lambda c: c["rerank_score"] or 0)
        # Strip any trailing whitespace but keep readability
        ctx = best["page_content"].strip()
        if len(ctx) > 8000:
            ctx = ctx[:8000].rsplit(" ", 1)[0] + " …"
        out.append({"id": f"f{g['id']}", "query": g["query"],
                     "ideal_context": ctx, "source": src})

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} faithfulness goldens to {OUT_PATH}")


if __name__ == "__main__":
    main()
