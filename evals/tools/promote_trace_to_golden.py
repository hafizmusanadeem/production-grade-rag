"""Promote a thumbs-down LangSmith trace to a human-reviewed draft golden."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
GOLDEN_DIR=ROOT/"evals"/"goldens"


def _query_from_run(run):
    inputs=run.inputs or {}
    for message in reversed(inputs.get("messages",[]) if isinstance(inputs,dict) else []):
        if isinstance(message,dict) and message.get("role")=="user": return message.get("content","")
    return inputs.get("query","") if isinstance(inputs,dict) else ""


def _sample_trace(client, project):
    for run in list(client.list_runs(project_name=project,is_root=True,run_type="chain")):
        tags=set(getattr(run,"tags",[]) or [])
        if "promoted" in tags: continue
        feedback=list(client.list_feedback(run_ids=[run.id]))
        if any(item.key=="user_rating" and item.score==0 for item in feedback):
            return run
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",choices=("retriever","correctness"),required=True)
    args=parser.parse_args()
    os.environ.setdefault("LANGSMITH_PROJECT","rag-production")
    from langsmith import Client
    client=Client(); project=os.getenv("LANGSMITH_PROJECT","rag-production")
    run=_sample_trace(client,project)
    if not run:
        print("No thumbs-down, unpromoted traces found."); return
    question=_query_from_run(run); answer=(run.outputs or {}).get("final_answer","")
    print(f"Trace: {str(run.id)}; question={question[:100]!r}; answer={answer[:100]!r}")
    print("Enter reviewed ideal answer (blank to skip):")
    ideal=input().strip()
    if not ideal: print("Skipping."); return
    print("Enter source file name:")
    source=input().strip()
    path=GOLDEN_DIR/("retriever.json" if args.target=="retriever" else "correctness.json")
    data=json.loads(path.read_text(encoding="utf-8"))
    prefix="k" if args.target=="retriever" else "c"
    entry_id=f"{prefix}{len(data)+1:03d}"
    entry={"id":entry_id,"query":question,"ideal_answer":ideal,"source":source,
           "status":"draft","tags":["from:trace"]}
    if args.target=="correctness":
        entry={"id":entry_id,"question":question,"ideal_answer":ideal,"source":source,
               "derived_from":[],"status":"draft","tags":["from:trace"]}
    data.append(entry)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Wrote draft {entry_id} to {path}. Human review is required before commit.")


if __name__=="__main__": main()
