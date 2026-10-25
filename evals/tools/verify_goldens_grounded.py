"""Audit golden answer tokens against exported corpus text."""
import csv
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DUMP=ROOT/"evals"/"artifacts"/"chunks_dump.json"
GOLDEN_DIR=ROOT/"evals"/"goldens"
OUT=ROOT/"evals"/"artifacts"/"grounding_report.csv"


def key_tokens(text):
    return set(re.findall(r'"[^"]+"|\b[A-Za-z][A-Za-z0-9_.-]{2,}\b|\b\d+(?:\.\d+)*\b',str(text)))


def main():
    if not DUMP.exists():
        print(f"Missing {DUMP}; run python -m evals.tools.export_chunks first.")
        raise SystemExit(1)
    chunks=json.loads(DUMP.read_text(encoding="utf-8"))
    corpus={}
    for chunk in chunks: corpus.setdefault(chunk.get("file_name",""),"")
    for chunk in chunks: corpus[chunk.get("file_name","")]+=" "+chunk.get("page_content","").lower()
    rows=[]; total=flagged=0
    for name in ("retriever.json","correctness.json","faithfulness.json"):
        path=GOLDEN_DIR/name
        if not path.exists(): continue
        for golden in json.loads(path.read_text(encoding="utf-8")):
            source=golden.get("source","")
            text=golden.get("ideal_answer","")
            if name=="correctness.json": text=golden.get("ideal_answer","")
            if name=="faithfulness.json": text=" ".join(golden.get("ideal_context",[]) if isinstance(golden.get("ideal_context"),list) else [golden.get("ideal_context","")])
            tokens=key_tokens(text); source_text=corpus.get(source,"")
            missing=[token for token in tokens if token.strip('"').lower() not in source_text]
            rate=len(missing)/len(tokens) if tokens else 0
            flagged_item=rate>.2; total+=1; flagged+=int(flagged_item)
            rows.append({"file":name,"id":golden.get("id"),"source":source,"token_count":len(tokens),
                         "ungrounded_count":len(missing),"ungrounded_rate":round(rate,4),
                         "flagged":flagged_item,"ungrounded_tokens":"; ".join(sorted(missing))})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["file","id","source","token_count","ungrounded_count","ungrounded_rate","flagged","ungrounded_tokens"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Grounding audit: {flagged}/{total} flagged; report: {OUT}")
    raise SystemExit(1 if total and flagged/total>.05 else 0)


if __name__=="__main__": main()
