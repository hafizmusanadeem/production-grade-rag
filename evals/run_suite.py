"""evals/run_suite.py -- the orchestrator.

Runs the FULL eval suite against ONE pipeline (built once and injected everywhere),
flattens every metric into dotted ids, stamps metadata, writes a snapshot JSON.
The snapshot is the atomic unit regression testing compares.

    python -m evals.run_suite                          -> baselines/candidate.json
    python -m evals.run_suite --baseline               -> baselines/baseline.json
    python -m evals.run_suite --label "k=10 + reranker"  (describe the change)
    python -m evals.run_suite --quiet                  -> suppress ops/safety chatter
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.generator import build_prompt
from evals.metric_registry import rule_for
from evals.eval_files import eval_retriever, eval_generator, eval_rag_pipeline, eval_application, eval_ops

load_dotenv(override=True)

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"

SUITE = ["retriever", "generator", "pipeline", "application", "safety", "ops"]


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _prompt_hash():
    return hashlib.sha256(str(build_prompt("test query", [], intent="technical")).encode()).hexdigest()[:12]


def build_metadata(label):
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "prompt_hash": _prompt_hash(),
        "label": label,
    }


def _slug(name):
    return name.strip().lower().replace(" ", "_")


def flatten_nested(namespace, summary):
    out = {}
    for metric, stats in summary.items():
        slug = _slug(metric)
        for stat, val in stats.items():
            out[f"{namespace}.{slug}.{stat}"] = val
    return out


def prefix_flat(namespace, flat):
    return {f"{namespace}.{k}": v for k, v in flat.items()}


def run_suite(label="", quiet=False, full=False):
    """Run all suites against one agent target; return a snapshot dict."""
    verbose = not quiet
    t0 = time.perf_counter()

    print("Building AgentTarget (once)...")
    agent_target = None  # we use eval_ops/eval_safety which create their own; or pass

    metrics = {}

    # retriever (uses RetrievalPipeline directly)
    print("\n[1] retriever eval...")
    metrics.update(flatten_nested("retriever", eval_retriever.run_local()))

    # generator (isolated, golden context)
    print("\n[2] generator eval...")
    metrics.update(flatten_nested("generator", eval_generator.run()))

    # RAG triad via AgentTarget
    print("\n[3] rag_pipeline (triad) eval...")
    metrics.update(flatten_nested("pipeline", eval_rag_pipeline.run(agent_target)))

    # application correctness/completeness via AgentTarget
    print("\n[4] application eval...")
    metrics.update(flatten_nested("application", eval_application.run(agent_target)))

    # safety
    print("\n[5] safety evals...")
    metrics.update(prefix_flat("safety", eval_ops_safety()))

    # ops
    print("\n[6] operational evals...")
    ops_snapshot = eval_ops.run_ops(verbose=verbose)
    metrics.update({f"ops.{k}": v for k, v in ops_snapshot.items()})

    elapsed = time.perf_counter() - t0

    if not full:
        metrics = {mid: v for mid, v in metrics.items() if rule_for(mid)["kind"] != "info"}

    # Stamp operational sub-keys under their proper namespace
    new_ops = {}
    for k, v in ops_snapshot.items():
        new_key = f"ops.{k}"
        new_ops[new_key] = v
    final = dict(metrics)
    final.update(new_ops)

    return {
        "metadata": {
            **build_metadata(label),
            "suite_seconds": round(elapsed, 1),
            "full": full,
            "n_metrics": len(metrics),
        },
        "metrics": final,
    }


def eval_ops_safety():
    """Wrapper to call the safety suite under the 'safety' namespace."""
    from evals.eval_files.eval_safety import run_safety
    return run_safety(verbose=True)


def write_snapshot(snapshot, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    return path


def print_snapshot(snapshot):
    meta = snapshot["metadata"]
    metrics = snapshot["metrics"]
    print("\n" + "=" * 74)
    print("SNAPSHOT")
    print("=" * 74)
    print(f"label       : {meta.get('label') or '(none)'}")
    print(f"created_at  : {meta['created_at']}")
    print(f"git_sha     : {meta['git_sha']}    prompt_hash: {meta['prompt_hash']}")
    print(f"suite_time  : {meta['suite_seconds']}s     metrics: {len(metrics)}")
    print("-" * 74)
    for key in sorted(metrics):
        val = metrics[key]
        shown = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {key:<44} {shown}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Run the full eval suite.")
    ap.add_argument("--baseline", action="store_true", help=f"write to {BASELINE_PATH}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--full", action="store_true", help="also write info metrics")
    args = ap.parse_args()

    out = args.out or (BASELINE_PATH if args.baseline else CANDIDATE_PATH)

    snapshot = run_suite(label=args.label, quiet=args.quiet, full=args.full)
    print_snapshot(snapshot)
    path = write_snapshot(snapshot, out)
    print(f"\nwrote snapshot -> {path}")


if __name__ == "__main__":
    main()
