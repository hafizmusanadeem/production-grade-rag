"""Build candidate faithfulness contexts from the exported corpus; human review required."""
import json
import re
from pathlib import Path

from evals.harness import select_goldens

ROOT=Path(__file__).resolve().parents[2]
DUMP_PATH=ROOT/"evals"/"artifacts"/"chunks_dump.json"
RETRIEVER_PATH=ROOT/"evals"/"goldens"/"retriever.json"
OUT_PATH=ROOT/"evals"/"goldens"/"faithfulness.json"


def tokens(text):
    return {t for t in re.findall(r"[a-zA-Z0-9_./-]+",str(text).lower())
            if len(t)>=3 or any(ch.isdigit() for ch in t) or "_" in t or "." in t}


def main():
    if not DUMP_PATH.exists():
        print(f"chunks_dump.json missing at {DUMP_PATH}. Run export_chunks first.")
        raise SystemExit(1)
    chunks=json.loads(DUMP_PATH.read_text(encoding="utf-8"))
    by_file={}
    for chunk in chunks: by_file.setdefault(chunk.get("file_name") or "unknown",[]).append(chunk)
    goldens=json.loads(RETRIEVER_PATH.read_text(encoding="utf-8"))
    selected=select_goldens(goldens,limit=30,seed=0,stratify_key="source")
    out=[]
    for golden in selected:
        candidates=by_file.get(golden["source"],[])
        ideal=tokens(golden.get("ideal_answer",""))
        scored=[]
        for chunk in candidates:
            content=chunk.get("page_content","")
            ct=tokens(content)
            score=len(ideal & ct)/len(ideal) if ideal else 0.0
            scored.append((score,chunk))
        scored.sort(key=lambda pair:pair[0],reverse=True)
        if not scored or scored[0][0] < .35:
            print(f"MANUAL REVIEW: {golden['id']} source={golden['source']} best_score={scored[0][0] if scored else 0:.3f}")
            continue
        chosen=[scored[0]]
        if len(scored)>1 and scored[1][0] >= .6*scored[0][0]:
            chosen.append(scored[1])
        out.append({"id":"f"+str(golden["id"])[1:],"query":golden["query"],
                    "ideal_context":[pair[1]["page_content"].strip() for pair in chosen],
                    "chunk_ids":[pair[1]["chunk_id"] for pair in chosen],"source":golden["source"]})
    OUT_PATH.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Wrote {len(out)} candidate faithfulness goldens to {OUT_PATH}. Human-review every item before committing.")


if __name__=="__main__": main()
