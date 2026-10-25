"""Run evaluation suites and write a regression snapshot."""
import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from evals.config import BASELINE_PATH, CANDIDATE_PATH, GOLDEN_DIR
from evals.metric_registry import rule_for
from evals.target import AgentTarget

load_dotenv(override=True)
SUITES = ["retriever", "generator", "pipeline", "application", "agent", "safety", "ops"]

# Suite name -> module under evals/eval_files (pipeline lives in eval_rag_pipeline).
SUITE_MODULES = {
    "retriever": "evals.eval_files.eval_retriever",
    "generator": "evals.eval_files.eval_generator",
    "pipeline": "evals.eval_files.eval_rag_pipeline",
    "application": "evals.eval_files.eval_application",
    "agent": "evals.eval_files.eval_agent",
    "safety": "evals.eval_files.eval_safety",
    "ops": "evals.eval_files.eval_ops",
}


def _git_sha():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"


def _git_dirty():
    try: return bool(subprocess.check_output(["git","status","--porcelain"],text=True,stderr=subprocess.DEVNULL).strip())
    except Exception: return None


def _golden_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GOLDEN_DIR.glob("*.json"))}


def _prompt_hash():
    try:
        from app.agents.prompts import prompt_fingerprint
        return prompt_fingerprint()
    except Exception: return "unknown"


def _json_safe(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,list): return [_json_safe(v) for v in value]
    return value


def build_metadata(label="", *, limit=None, suite_errors=None):
    from app.config import RERANK_TOP_N, RETRIEVAL_LIMIT_K, settings
    try:
        from app.ingestion.manifest import corpus_fingerprint
        fingerprint = corpus_fingerprint("DATA")
    except Exception: fingerprint = "unavailable"
    return {"created_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha":_git_sha(),"dirty":_git_dirty(),"prompt_hash":_prompt_hash(),"label":label,
            "limit":limit,"judge_model":os.getenv("EVAL_JUDGE_MODEL","openai/gpt-oss-20b:free"),
            "planner_model":f"@{settings.PORTKEY_SLUG_2}/openrouter/free",
            "responder_config_id":os.getenv("PORTKEY_CONFIG_ID","unknown"),
            "golden_hashes":_golden_hashes(),"corpus_fingerprint":fingerprint,
            "QDRANT_COLLECTION":settings.QDRANT_COLLECTION,
            "settings":{"chunk_size":settings.CHUNK_SIZE,"chunk_overlap":settings.CHUNK_OVERLAP,
                        "embedding_model":settings.GEMINI_EMBEDDING_MODEL,"retrieval_limit_k":RETRIEVAL_LIMIT_K,
                        "rerank_top_n":RERANK_TOP_N,"rerank_min_score":settings.RERANK_MIN_SCORE},
            "suite_errors":suite_errors or {}}


def prefix_flat(namespace, values): return {f"{namespace}.{k}":v for k,v in values.items()}


def resolved_models(results):
    """JSON-safe sets of the models actually observed on the shared target (D-8).

    More than one value means a gateway fallback fired mid-run, so the snapshot
    is not apples-to-apples.
    """
    planners, responders = set(), set()
    for r in results:
        usage = getattr(r, "usage", None) or {}
        planners.add(str(usage.get("planner_model") or "unknown"))
        responders.add(str(usage.get("responder_model") or "unknown"))
    return {"resolved_planner_models": sorted(planners),
            "resolved_responder_models": sorted(responders)}


def warn_on_multi_model(metadata):
    for key in ("resolved_planner_models", "resolved_responder_models"):
        values = metadata.get(key) or []
        if len(values) > 1:
            print(f"WARNING: {key} saw more than one value ({', '.join(values)}); "
                  "a gateway fallback fired and the snapshot is not apples-to-apples.")


def run_suite(label="", quiet=False, full=False, *, limit=None, only=None):
    started=time.perf_counter(); target=AgentTarget(); metrics={}; errors={}
    for name in only or SUITES:
        if name not in SUITES:
            errors[name]="unknown suite"; continue
        module_name=SUITE_MODULES[name]
        try: module=importlib.import_module(module_name)
        except ImportError:
            errors[name]="module not implemented"; print(f"skipping {name}: not implemented"); continue
        try:
            if name=="ops": flat=module.run_ops(target,verbose=not quiet)
            elif name=="retriever": flat=module.run(target,variant="production",limit=limit,verbose=not quiet)
            else:
                if not hasattr(module,"run"):
                    errors[name]="no run()"; print(f"skipping {name}: not implemented"); continue
                flat=module.run(target,limit=limit,verbose=not quiet)
            if not isinstance(flat,dict): raise TypeError("suite run() must return a dict")
            metrics.update(prefix_flat(getattr(module,"NAMESPACE",name),flat))
        except Exception as exc:
            errors[name]=f"{type(exc).__name__}: {exc}"
            print(f"suite {name} failed: {errors[name]}")
    if not full: metrics={k:v for k,v in metrics.items() if rule_for(k)["kind"]!="info"}
    metadata=build_metadata(label,limit=limit,suite_errors=errors)
    metadata.update({"suite_seconds":round(time.perf_counter()-started,1),"full":full,"n_metrics":len(metrics)})
    metadata.update(resolved_models(target.results))
    warn_on_multi_model(metadata)
    return {"metadata":metadata,"metrics":_json_safe(metrics)}


def write_snapshot(snapshot,path):
    os.makedirs(os.path.dirname(str(path)) or ".",exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(_json_safe(snapshot),f,indent=2,sort_keys=True,allow_nan=False)
    return path


def print_snapshot(snapshot):
    m=snapshot["metadata"]; print("\\nSNAPSHOT")
    print(f"label={m.get('label') or '(none)'} git_sha={m['git_sha']} prompt_hash={m['prompt_hash']}")
    for k,v in sorted(snapshot["metrics"].items()): print(f"  {k:<44} {v}")
    if m.get("suite_errors"): print("Suite errors:",m["suite_errors"])


def main():
    ap=argparse.ArgumentParser(description="Run evaluation suites.")
    ap.add_argument("--baseline",action="store_true"); ap.add_argument("--out",default=None)
    ap.add_argument("--label",default=""); ap.add_argument("--only",default=None)
    ap.add_argument("--limit",type=int,default=None); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--quiet",action="store_true"); ap.add_argument("--full",action="store_true")
    args=ap.parse_args(); limit=10 if args.smoke else args.limit
    out=args.out or str(BASELINE_PATH if args.baseline else CANDIDATE_PATH)
    if args.baseline and limit is not None and "baseline_smoke" not in out:
        ap.error("Refusing a limited run as full baseline; use --out baselines/baseline_smoke.json")
    only=[x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    snapshot=run_suite(args.label,args.quiet,args.full,limit=limit,only=only)
    print_snapshot(snapshot); write_snapshot(snapshot,out); print(f"wrote snapshot -> {out}")
    return 1 if snapshot["metadata"]["suite_errors"] else 0


if __name__=="__main__": raise SystemExit(main())
