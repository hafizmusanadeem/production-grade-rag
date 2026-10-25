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

from evals.config import BASELINE_PATH, CANDIDATE_PATH
from evals.metric_registry import rule_for

EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


def load(path):
    with open(path) as f:
        return json.load(f)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and not (isinstance(x, float) and math.isnan(x))


def classify(mid, bv, cv, rule):
    if isinstance(bv, float) and math.isnan(bv): bv = None
    if isinstance(cv, float) and math.isnan(cv): cv = None
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

    if any(r["status"] == "blocked" or (r["status"] == "dropped" and r["kind"] == "gate") for r in rows):
        verdict = "FAIL"
    elif any(r["status"] == "regressed" or (r["status"] == "dropped" and r["kind"] == "guardrail") for r in rows):
        verdict = "REVIEW"
    elif candidate.get("metadata", {}).get("suite_errors"):
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


# Rows that never belong in a PR comment: they say nothing happened.
_MARKDOWN_SHOW = ("blocked", "regressed", "dropped", "new")


def _delta_str(row):
    d = row["delta"]
    if isinstance(d, (int, float)) and not isinstance(d, bool):
        return f"{d:+.4g}"
    return ""


def markdown_report(verdict, rows, baseline_meta, candidate_meta, baseline_path, candidate_path):
    """GitHub-flavoured markdown for a PR comment (C2)."""
    def short(path):
        return str(path).replace("\\", "/").rsplit("/", 1)[-1]

    lines = ["<!-- evals-compare -->", "", "## Evals compare", ""]
    lines.append("| | baseline | candidate |")
    lines.append("|---|---|---|")
    lines.append(f"| snapshot | `{short(baseline_path)}` | `{short(candidate_path)}` |")
    lines.append(f"| git | `{baseline_meta.get('git_sha', '?')}` | "
                 f"`{candidate_meta.get('git_sha', '?')}` |")
    lines.append(f"| prompt | `{baseline_meta.get('prompt_hash', '?')}` | "
                 f"`{candidate_meta.get('prompt_hash', '?')}` |")
    lines.append("")
    lines.append(f"**Verdict: {verdict}** (exit {EXIT_CODE[verdict]})")
    lines.append("")

    interesting = [r for r in rows if r["status"] in _MARKDOWN_SHOW]
    guarded = [r for r in interesting if r["kind"] in ("gate", "guardrail")]
    other = [r for r in interesting if r["kind"] == "info"]
    if guarded:
        lines.append(f"### Gate / guardrail rows that are not flat ({len(guarded)})")
        lines.append("")
        lines.append("| metric | baseline | candidate | delta | kind | status |")
        lines.append("|---|---:|---:|---:|---|---|")
        for r in guarded:
            lines.append(f"| `{r['id']}` | {_fmt(r['baseline'])} | {_fmt(r['candidate'])} | "
                         f"{_delta_str(r)} | {r['kind']} | **{r['status']}** |")
        lines.append("")
    else:
        lines.append("No gate or guardrail row changed. Nothing to review.")
        lines.append("")

    errors = candidate_meta.get("suite_errors") or {}
    if errors:
        lines.append("### Suite errors")
        lines.append("")
        for suite, error in sorted(errors.items()):
            lines.append(f"- `{suite}`: {error}")
        lines.append("")
    if other:
        lines.append("<details>")
        lines.append(f"<summary>Info rows changed ({len(other)})</summary>")
        lines.append("")
        lines.append("| metric | baseline | candidate | delta |")
        lines.append("|---|---:|---:|---:|")
        for r in other:
            lines.append(f"| `{r['id']}` | {_fmt(r['baseline'])} | {_fmt(r['candidate'])} | "
                         f"{_delta_str(r)} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compare two eval snapshots.")
    ap.add_argument("--baseline", default=BASELINE_PATH)
    ap.add_argument("--candidate", default=CANDIDATE_PATH)
    ap.add_argument("--all", action="store_true", help="also show info metrics")
    ap.add_argument("--markdown", action="store_true",
                    help="emit GitHub-flavoured markdown only (for a PR comment)")
    args = ap.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)

    bl = baseline.get("metadata", {})
    cl = candidate.get("metadata", {})
    verdict, rows = compare(baseline, candidate)

    if args.markdown:
        print(markdown_report(verdict, rows, bl, cl, args.baseline, args.candidate))
        sys.exit(EXIT_CODE[verdict])

    for key in ("golden_hashes", "corpus_fingerprint", "judge_model", "limit", "QDRANT_COLLECTION",
                "resolved_planner_models", "resolved_responder_models"):
        if bl.get(key) != cl.get(key):
            print(f"WARNING: snapshot metadata differs for {key}: baseline={bl.get(key)!r}, candidate={cl.get(key)!r}")
    for suite, error in (cl.get("suite_errors") or {}).items():
        print(f"WARNING: suite {suite} failed: {error}")
    label_b = bl.get("label") or args.baseline
    label_c = cl.get("label") or args.candidate
    git_b = bl.get("git_sha", "?")
    prompt_b = bl.get("prompt_hash", "?")
    print(f"baseline : {label_b}   ({git_b}, prompt {prompt_b})")
    print(f"candidate: {label_c}   ({cl.get('git_sha', '?')}, prompt {cl.get('prompt_hash', '?')})")

    print_report(verdict, rows, show_all=args.all)
    sys.exit(EXIT_CODE[verdict])


if __name__ == "__main__":
    main()
