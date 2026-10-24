"""evals/metric_registry.py -- the metric registry.

For every metric the suite produces, three questions decide how a change is
judged: direction, kind (gate/guardrail/info), tolerance. The keys in this file
are NOT public API --- rule_for(metric_id) is the single entry point evals and
compare.py use. Do not change the rules here unless you also update P8.

Reading this file: judge metrics are gated on their AVERAGE SCORE (0-1 scale)
not pass_rate -- see docstring at top. Operational metrics are direct
measurements (percentiles, cost, rates) with unchanged rules.
"""
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "goldens"

# Presets ----------------------------------------------------------
# tol     = absolute tolerance, in the metric's own units
# rel_tol = relative tolerance, as a fraction of |baseline|
# A regression counts only if the worsening exceeds max(tol, rel_tol * |baseline|).

# Gate on average score (0-1): any drop BLOCKS the release. Safety lives here.
GATE_HIGHER_AVG = {"direction": "higher", "kind": "gate",      "tol": 0.02, "rel_tol": 0.0}
GATE_LOWER_AVG  = {"direction": "lower",  "kind": "gate",      "tol": 0.02, "rel_tol": 0.0}
QUALITY_GUARD   = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}   # quality

LATENCY_GUARD   = {"direction": "lower",  "kind": "guardrail", "tol": 0.0,  "rel_tol": 0.25}   # 25%
COST_GUARD      = {"direction": "lower",  "kind": "guardrail", "tol": 1e-6, "rel_tol": 0.15}  # 15% (abs floor)
SUCCESS_GUARD   = {"direction": "higher", "kind": "guardrail", "tol": 1.0,  "rel_tol": 0.0}
ERROR_GUARD     = {"direction": "lower",  "kind": "guardrail", "tol": 1.0,  "rel_tol": 0.0}
SLO_BOOL_GUARD  = {"direction": "higher", "kind": "guardrail", "tol": 0.0,  "rel_tol": 0.0, "bool": True}
INFO            = {"direction": "higher", "kind": "info",      "tol": 0.0,  "rel_tol": 0.0}

# The operational number that drives the verdict. Everything else under ops.*
# (p50/p99/mean, token counts, monthly total, INR, request count) is info.
LATENCY_GUARDED = {"ops.latency.e2e_p95_ms"}


def rule_for(metric_id: str) -> dict:
    mid = metric_id

    # ---- SAFETY GATES (on average score) ----
    if mid == "safety.toxicity.avg_toxicity":
        return GATE_LOWER_AVG
    if mid.startswith("safety.") and mid.endswith("avg_score"):
        return GATE_HIGHER_AVG

    # ---- QUALITY GUARDRAILS (on avg score) ----
    if mid.endswith("avg_score"):
        return QUALITY_GUARD

    # ---- OPERATIONAL ----
    if mid in LATENCY_GUARDED:
        return LATENCY_GUARD
    if mid == "ops.cost.cost_per_query_usd":
        return COST_GUARD
    if mid == "ops.reliability.success_rate":
        return SUCCESS_GUARD
    if mid == "ops.reliability.error_rate":
        return ERROR_GUARD
    if mid.endswith("_pass"):
        return SLO_BOOL_GUARD

    return INFO
