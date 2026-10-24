"""evals/compare.py -- the decision function.

Reads two snapshots (baseline, candidate) written by run_suite.py, looks up each
metric's rule in the registry, classifies every metric, returns PASS/REVIEW/FAIL.
Exit code makes it a CI gate: PASS=0, FAIL=1, REVIEW=2.
"""
import argparse
import json
import math
import sys
from collections import Counter

from evals.metric_registry import rule_for

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"

EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


def load(path):
    with open(path) as f:
        return json.load(f)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and not (isinstance(x, float) and math.isnan(x))


def classify(mid, bv, cv, rule):
    row = {"id": mid, "baseline": bv, "candidate": cv,
           "kind": rule["kind"], "direction": rule["direction"],
           "delta": None, "status": None}

    if bv is None:
        row["status"] = "new"
        return row
    if cv is None:
        row["status"] = "dropped"
        return row

    if rule["kind"] == "info":
        if _is_number(bv) and _is_number(cv):
            row["delta"] = cv - bv
        row["status"] = "info"
        return row

    # boolean SLO flags
    if rule.get("bool") or isinstance(bv, bool) or isinstance(cv, bool):
        if bv == cv:
            row["status"] = "flat"
        elif (not bv) and cv:
            row["status"] = "improved"
        else:
            row["status"] = ("blocked" if rule["kind"] == "gate" else "regressed")
        return row

    if not (_is_number(bv) and _is_number(cv)):
        row["status"] = "info"
        return row

    delta = cv - bv
    row["delta"] = delta
    higher_better = rule["direction"] == "higher"
    improved = (delta > 0) if higher_better else (delta < 0)
    if improved:
        row["status"] = "improved"
        return row

    worse = abs(delta)
    tolerance = max(rule["tol"], rule["rel_tol"] * abs(bv))
    if worse <= tolerance:
        row["status"] = "flat"
    else:
        row["status"] = ("blocked" if rule["kind"] == "gate" else "regressed")
    return row


def compare(baseline, candidate):
    b = baseline.get("metrics", {})
    c = candidate.get("metrics", {})

    rows = [classify(mid, b.get(mid), c.get(mid), rule_for(mid))
            for mid in sorted(set(b) | set(c))]

    if any(r["status"] == "blocked" for r in rows):
        verdict = "FAIL"
    elif any(r["status"] == "regressed" for r in rows):
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    return verdict, rows


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def print_report(verdict, rows, show_all=False):
    order = {"blocked": 0, "regressed": 1, "dropped": 2, "new": 3,
             "improved": 4, "flat": 5, "info": 6}
    shown = [r for r in rows if show_all or r["kind"] != "info"]
    shown.sort(key=lambda r: (order.get(r["status"], 9), r["id"]))

    print("=" * 96)
    print(f"{'metric':<46} {'baseline':>10} {'candidate':>10} {'delta':>10}  status")
    print("-" * 96)
    for r in shown:
        d = r["delta"]
        dstr = f"{d:+.4g}" if isinstance(d, (int, float)) and not isinstance(d, bool) else ""
        marker = "  <<<" if r["status"] in ("blocked", "regressed") else ""
        print(f"{r['id']:<46} {_fmt(r['baseline']):>10} {_fmt(r['candidate']):>10} "
              f"{dstr:>10}  {r['status']}{marker}")

    counts = Counter(r["status"] for r in rows)
    line = "  ".join(f"{k}={counts[k]}" for k in
                     ["blocked", "regressed", "improved", "flat", "new", "dropped", "info"]
                     if counts[k])
    print("-" * 96)
    print(line)
    info_n = counts["info"]
    if not show_all and info_n:
        print(f"({info_n} info metrics hidden; pass --all to show them)")
    print("=" * 96)

    banner = {
        "PASS":   "PASS   -- no gate blocked, no guardrail regressed. Safe to promote.",
        "REVIEW": "REVIEW -- a guardrail regressed beyond tolerance. A human decides.",
        "FAIL":   "FAIL   -- a gate regressed. Blocked.",
    }[verdict]
    print(f"VERDICT: {banner}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="Compare two eval snapshots.")
    ap.add_argument("--baseline", default=BASELINE_PATH)
    ap.add_argument("--candidate", default=CANDIDATE_PATH)
    ap.add_argument("--all", action="store_true", help="also show info metrics")
    args = ap.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)

    bl = baseline.get("metadata", {})
    cl = candidate.get("metadata", {})
    label_b = bl.get("label") or args.baseline
    label_c = cl.get("label") or args.candidate
    git_b = bl.get("git_sha", "?")
    prompt_b = bl.get("prompt_hash", "?")
    print(f"baseline : {label_b}   ({git_b}, prompt {prompt_b})")
    print(f"candidate: {label_c}   ({cl.get('git_sha', '?')}, prompt {cl.get('prompt_hash', '?')})")

    verdict, rows = compare(baseline, candidate)
    print_report(verdict, rows, show_all=args.all)
    sys.exit(EXIT_CODE[verdict])


if __name__ == "__main__":
    main()
