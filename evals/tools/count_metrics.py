"""Count the non-info metrics a snapshot would actually gate on (definition of done)."""
import argparse
import collections
import json

from evals.config import CANDIDATE_PATH
from evals.metric_registry import rule_for

TARGET = 60


def summarise(snapshot, target=TARGET):
    """Group non-info metrics by namespace; suites that crashed contribute nothing."""
    metrics = snapshot.get("metrics", {})
    per_namespace = collections.defaultdict(int)
    for mid in metrics:
        if rule_for(mid)["kind"] != "info":
            per_namespace[mid.split(".", 1)[0]] += 1
    errors = snapshot.get("metadata", {}).get("suite_errors") or {}
    return {"per_namespace": dict(sorted(per_namespace.items())),
            "non_info": sum(per_namespace.values()),
            "total": len(metrics),
            "target": target,
            "meets_target": sum(per_namespace.values()) >= target,
            "suite_errors": errors}


def render(summary):
    lines = [f"{'namespace':<14} {'gated metrics':>14}", "-" * 30]
    for namespace, count in summary["per_namespace"].items():
        lines.append(f"{namespace:<14} {count:>14}")
    lines.append("-" * 30)
    lines.append(f"non-info: {summary['non_info']}   info+other: "
                 f"{summary['total'] - summary['non_info']}   target: {summary['target']}")
    if summary["suite_errors"]:
        for suite, error in summary["suite_errors"].items():
            lines.append(f"suite {suite} FAILED: {error}")
    lines.append("PASS: target met" if summary["meets_target"]
                 else "FAIL: below target -- a crashed or missing suite is the usual cause")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Count gated metrics in a snapshot.")
    ap.add_argument("--snapshot", default=str(CANDIDATE_PATH))
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()
    with open(args.snapshot, encoding="utf-8") as f:
        snapshot = json.load(f)
    summary = summarise(snapshot, args.target)
    print(render(summary))
    return 0 if summary["meets_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
