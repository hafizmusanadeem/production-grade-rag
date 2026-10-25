"""Operational metrics measured from AgentTarget's real result metadata."""
import math
from statistics import mean

from evals.config import GOLDEN_DIR, SLO_E2E_P95_MS
from evals.harness import load_goldens
from evals.target import AgentTarget

NAMESPACE = "ops"
QUESTIONS_PATH = GOLDEN_DIR / "ops_questions.json"


def load_questions(path=QUESTIONS_PATH):
    """Representative ops questions; falls back to a built-in list if absent."""
    try:
        items = load_goldens(path)
        return [item["question"] for item in items if item.get("question")]
    except (FileNotFoundError, KeyError):
        return list(_FALLBACK_QUESTIONS)


_FALLBACK_QUESTIONS = [
 "Which restart policies can a Kubernetes Job Pod use, and which one is not allowed?",
 "What is the difference between Indexed and NonIndexed completion modes?",
 "How is the database-backup CronJob configured? Include its schedule, timezone, retries, timeout, and job history.",
 "How do I exclude a specific container from VPA scaling?",
]

QUESTIONS = load_questions()


def percentile(values, p):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values: return float("nan")
    if len(values) == 1: return values[0]
    pos = (len(values)-1)*p/100; lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo]*(hi-pos)+values[hi]*(pos-lo)


def lat_summary(samples):
    values = [v for v in samples if v is not None and math.isfinite(v)]
    if not values:
        n = float("nan")
        return {"n": 0, "mean": n, "p50": n, "p95": n, "p99": n, "min": n, "max": n}
    return {"n": len(values), "mean": mean(values), "p50": percentile(values,50),
            "p95": percentile(values,95), "p99": percentile(values,99), "min": min(values), "max": max(values)}


def run_latency(at, verbose=True):
    for q in QUESTIONS[:3]: at.invoke(q)  # warm-up, discarded
    stages = {k: [] for k in ("e2e","planner","embed_qdrant","rerank","generate")}
    for q in QUESTIONS:
        result = at.invoke(q)
        if result.error: continue
        for stage in stages:
            key = "total" if stage == "e2e" else stage
            if result.timings_ms.get(key) is not None: stages[stage].append(result.timings_ms[key])
    summaries = {k: lat_summary(v) for k,v in stages.items()}
    out = {f"{stage}_{stat}_ms": summaries[stage][stat] for stage in stages for stat in ("mean","p50","p95","p99")}
    e2e = summaries["e2e"]
    out["slo_e2e_pass"] = math.isfinite(e2e["p95"]) and e2e["p95"] <= SLO_E2E_P95_MS
    means = [summaries[k]["mean"] for k in ("planner","embed_qdrant","rerank","generate")]
    out["stage_sum_ok"] = bool(math.isfinite(e2e["mean"]) and e2e["mean"] > 0 and all(math.isfinite(x) for x in means)
                               and abs(sum(means)-e2e["mean"])/e2e["mean"] < .10)
    if verbose: print("[latency]", {k:v for k,v in out.items() if "p95" in k or k.endswith("_pass")})
    return out


def run_cost(at, verbose=True):
    from evals.pricing import shadow_cost
    costs, inputs, outputs, cached = [], [], [], []
    for q in QUESTIONS:
        result = at.invoke(q)
        if result.error or result.declined: continue
        usage = result.usage or {}
        inp = int(usage.get("prompt_tokens",0)) + int(usage.get("planner_prompt_tokens",0))
        out = int(usage.get("completion_tokens",0)) + int(usage.get("planner_completion_tokens",0))
        cache = int(usage.get("cached_tokens",0))
        inputs.append(inp); outputs.append(out); cached.append(cache)
        price = shadow_cost(inp, out, cache)
        if price is not None: costs.append(price)
    avg = mean(costs) if costs else None
    return {"cost_per_query_usd": avg, "cost_per_query_inr": avg*88 if avg is not None else None,
            "avg_input_tokens": mean(inputs) if inputs else 0, "avg_output_tokens": mean(outputs) if outputs else 0,
            "avg_cached_tokens": mean(cached) if cached else 0}


def run_reliability(at, verbose=True):
    total = success = 0; errors = {}
    for q in QUESTIONS:
        result = at.invoke(q); total += 1
        if not result.error and result.answer: success += 1
        else:
            code = result.error or "empty_answer"; errors[code] = errors.get(code,0)+1
    out = {"total_requests": total, "success_rate": 100*success/total if total else 0,
           "error_rate": 100*(total-success)/total if total else 0}
    for code in ("retrieval_unavailable","rerank_failed","llm_unavailable","llm_empty_response","internal_error","empty_answer"):
        out[f"error_{code}_count"] = errors.get(code,0)
    return out


def run_ops(at=None, verbose=True):
    at = at or AgentTarget(); out = {}
    out.update({f"latency.{k}":v for k,v in run_latency(at,verbose).items()})
    out.update({f"cost.{k}":v for k,v in run_cost(at,verbose).items()})
    out.update({f"reliability.{k}":v for k,v in run_reliability(at,verbose).items()})
    return out


def run(target=None, *, limit=None, ids=None, verbose=True):
    return run_ops(target, verbose)
