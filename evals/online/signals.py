"""Live health signals from LangSmith over a rolling window.

Exit codes: 0 = every signal inside its threshold, 1 = at least one exceeded.
Thresholds are placeholders calibrated by a human after the first week of traffic.
"""
import argparse
import math
import os
import statistics
from datetime import datetime, timedelta, timezone

# TODO(human): calibrate from the first week of production traffic.
THRESHOLDS = {
    "decline_rate": 0.35,
    "no_relevant_context_rate": 0.35,
    "error_rate": 0.02,
    "thumbs_down_rate": 0.25,
    "faithfulness_mean": 0.70,
    "answer_relevancy_mean": 0.70,
    "empty_answer_rate": 0.02,
}
P95_SLO_MULTIPLE = 1.5  # e2e p95 above slo_ms * this is an alert


def _parse_ts(value):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _plan_of(run):
    plan = getattr(run, "plan", None) or []
    if isinstance(plan, str):
        return [plan]
    return list(plan)


def _rating_of(run):
    score = getattr(run, "feedback_stats", None)
    if isinstance(score, dict):
        if score.get("user_rating") is not None:
            return int(score["user_rating"])
        for key, value in sorted(score.items()):
            if "rating" in key.lower() and value is not None:
                return int(value)
    return None


def _answer_of(run):
    outputs = getattr(run, "outputs", None) or {}
    answer = outputs.get("final_answer") or outputs.get("answer") or ""
    return str(answer)


def compute_signals(runs, *, slo_ms=3000.0):
    """Signals for a list of run-like objects (duck-typed, testable offline).

    A run exposes: .plan (list[str]), .error, .outputs (dict),
    .start_time/.end_time, and optionally .feedback_stats / .faithfulness /
    .answer_relevancy.
    """
    total = 0
    decline = no_ctx = errors = empty = 0
    latencies, faithfulness, relevancy = [], [], []
    rated, negative = 0, 0

    for run in runs:
        total += 1
        plan = _plan_of(run)
        if any("Context: none relevant" in str(step) for step in plan):
            no_ctx += 1
        if getattr(run, "error", None):
            errors += 1
        elif not _answer_of(run).strip():
            empty += 1
        start, end = _parse_ts(getattr(run, "start_time", None)), _parse_ts(getattr(run, "end_time", None))
        if start is not None and end is not None and end > start:
            latencies.append((end - start) * 1000.0)
        rating = _rating_of(run)
        if rating is not None:
            rated += 1
            if rating == 0:
                negative += 1
        faith = getattr(run, "faithfulness", None)
        if faith is not None:
            faithfulness.append(float(faith))
        rel = getattr(run, "answer_relevancy", None)
        if rel is not None:
            relevancy.append(float(rel))

    def rate(count):
        return count / total if total else 0.0

    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else (
        latencies[0] if latencies else float("nan"))

    signals = {
        "total_runs": total,
        "decline_rate": rate(no_ctx),  # declines are the no-relevant-context branch
        "no_relevant_context_rate": rate(no_ctx),
        "empty_answer_rate": rate(empty),
        "error_rate": rate(errors),
        "rated_runs": rated,
        "thumbs_down_rate": negative / rated if rated else 0.0,
        "faithfulness_mean": statistics.mean(faithfulness) if faithfulness else float("nan"),
        "answer_relevancy_mean": statistics.mean(relevancy) if relevancy else float("nan"),
        "e2e_p95_ms": p95,
    }
    return signals


def evaluate(signals, thresholds=None, *, slo_ms=3000.0):
    """Mark each signal ok/alert. Returns {name: (value, ok)}."""
    thresholds = thresholds or THRESHOLDS
    out = {}
    for name, limit in thresholds.items():
        value = signals.get(name, float("nan"))
        if isinstance(value, float) and math.isnan(value):
            out[name] = (value, True)
            continue
        if name in ("faithfulness_mean", "answer_relevancy_mean"):
            out[name] = (value, value >= limit)
        else:
            out[name] = (value, value <= limit)
    p95 = signals.get("e2e_p95_ms", float("nan"))
    out["e2e_p95_ms"] = (p95, (isinstance(p95, float) and math.isnan(p95))
                         or p95 <= slo_ms * P95_SLO_MULTIPLE)
    return out


def render(signals, marks, slo_ms=3000.0):
    lines = [f"{'signal':<26} {'value':>12}  {'threshold':>12}  status", "-" * 66]
    for name, (value, ok) in marks.items():
        if name == "e2e_p95_ms":
            limit = f"<= {slo_ms * P95_SLO_MULTIPLE:g}"
        elif name in THRESHOLDS:
            if name in ("faithfulness_mean", "answer_relevancy_mean"):
                limit = f">= {THRESHOLDS[name]:g}"
            else:
                limit = f"<= {THRESHOLDS[name]:g}"
        else:
            limit = ""
        lines.append(f"{name:<26} {value:>12.4g}  {limit:>12}  {'ok' if ok else 'ALERT'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compute online health signals.")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--slo-ms", type=float, default=3000.0)
    args = ap.parse_args()

    try:  # imports are namespaced so `import evals.online.signals` works offline
        from langsmith import Client
        client = Client()
        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
        project = os.getenv("LANGSMITH_PROJECT", "rag-production-grade")
        runs = list(client.list_runs(start_time=since, project_name=project))
    except Exception as exc:
        print(f"Could not list LangSmith runs: {type(exc).__name__}: {exc}")
        print("Check LANGSMITH_API_KEY; this tool is a no-op without it.")
        return 1

    signals = compute_signals(runs, slo_ms=args.slo_ms)
    marks = evaluate(signals, slo_ms=args.slo_ms)
    print(f"Reports over the last {args.hours:g}h (window ending now)\n")
    print(render(signals, marks))
    return 0 if all(ok for _, ok in marks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
