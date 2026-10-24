"""evals/eval_files/eval_ops.py -- operational evals: LATENCY, COST, RELIABILITY.

Merged into one file so run_ops() returns a single snapshot dict (the ops slice of
the regression snapshot). Each sub-eval exposes run_*() that PRINTS and RETURNS a
flat metrics dict. Latency uses percentiles against SLO budgets; cost is derived
(tokens x price); reliability tracks success/error/retry rates with backoff.
"""
import math
import time
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCaseParams

from app.generator import generate
from app.gateway import portkey_client
from evals.config import SLO_E2E_P95_MS, SLO_TTFT_P95_MS
from evals.target import AgentTarget

load_dotenv(override=True)

QUESTIONS = [
    "Which restart policies can a Kubernetes Job Pod use, and which one is not allowed?",
    "What is the difference between Indexed and NonIndexed completion modes?",
    "How is the database-backup CronJob configured? Include its schedule, timezone, retries, timeout, and job history.",
    "How do I exclude a specific container from VPA scaling?",
]


# ============================================================
# LATENCY
# ============================================================
LAT_REPEATS = 5
WARMUP = 2
SLO_P95_MS = SLO_E2E_P95_MS       # 3s
SLO_TTFT_P95_MS = SLO_TTFT_P95_MS  # 1.2s


def percentile(values, p):
    clean = [v for v in values if v is not None]
    if not clean:
        return float("nan")
    s = sorted(clean)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def lat_benchmark(at: AgentTarget):
    print("[latency] warming up (%d runs, discarded)..." % WARMUP)
    for i in range(WARMUP):
        at.invoke(QUESTIONS[i % len(QUESTIONS)])

    total, retrieval, generation, ttft, lens = [], [], [], [], []
    print("[latency] measuring...")
    for q in QUESTIONS:
        for _ in range(LAT_REPEATS):
            t0 = time.perf_counter()
            r = at.invoke(q)
            t1 = time.perf_counter()
            total.append((t1 - t0) * 1000)
            lens.append(len(r.answer or ""))

    return {"total": total, "retrieval": retrieval, "generation": generation,
            "ttft": ttft, "answer_len": lens}


def lat_summary(samples):
    clean = [s for s in samples if s is not None]
    return {
        "n": len(clean), "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50), "p95": percentile(clean, 95),
        "p99": percentile(clean, 99), "min": min(clean), "max": max(clean),
    }


def run_latency(at: AgentTarget, verbose=True):
    results = lat_benchmark(at)
    if verbose:
        print("\n" + "=" * 78)
        print("LATENCY (milliseconds)")
        print("=" * 78)
        for label, s in [("end-to-end", lat_summary(results["total"])),
                         ("generation", lat_summary(results["generation"])),
                         ("retrieval", lat_summary(results["retrieval"]))]:
            v = f"{s['p95']:7.0f}" if not math.isnan(s["p95"]) else "nan"
            print(f"{label:<12} p95={v}")
        total = lat_summary(results["total"])
        print("-" * 78)
        print(f"avg answer length: {sum(results['answer_len']) / len(results['answer_len']):.0f}")
    total = lat_summary(results["total"])
    metrics = {
        "e2e_mean_ms": total["mean"], "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"], "e2e_p99_ms": total["p99"],
        "slo_e2e_pass": total["p95"] <= SLO_P95_MS,
    }
    if results["ttft"]:
        ttft = lat_summary(results["ttft"])
        metrics["ttft_p95_ms"] = ttft["p95"]
        metrics["slo_ttft_pass"] = ttft["p95"] <= SLO_TTFT_P95_MS
    if results["retrieval"]:
        metrics["retrieval_p95_ms"] = lat_summary(results["retrieval"])["p95"]
        metrics["generation_p95_ms"] = lat_summary(results["generation"])["p95"]
    return metrics


# ============================================================
# COST
# ============================================================
COST_REPEATS = 3
PRICE_INPUT = 0.15 / 1_000_000
PRICE_CACHED = 0.075 / 1_000_000
PRICE_OUTPUT = 0.60 / 1_000_000
QUERIES_PER_DAY = 2000
USD_TO_INR = 88.0
COST_BUDGET_PER_QUERY = 0.0015


def cost_measure(at: AgentTarget):
    docs = at.invoke(QUESTIONS[0]).context  # representative retrieved context
    ctx = "\n\n".join(docs)
    prompt = generate(QUESTIONS[0], docs)     # same prompt + llm, real context
    resp = portkey_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], temperature=0.1)
    usage = getattr(resp, "usage", None) or {}
    inp = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    details = usage.get("input_token_details") or {}
    cached = details.get("cache_read", 0) or 0
    return {"input": inp, "output": out, "cached": cached}


def cost_usd(tok):
    uncached = max(tok["input"] - tok["cached"], 0)
    c_in = uncached / 1_000_000 * PRICE_INPUT
    c_c = tok["cached"] / 1_000_000 * PRICE_CACHED
    c_o = tok["output"] / 1_000_000 * PRICE_OUTPUT
    return {"input": c_in, "cached": c_c, "output": c_o, "total": c_in + c_c + c_o}


def run_cost(at: AgentTarget, verbose=True):
    rows = []
    print("[cost] measuring token usage...")
    for q in QUESTIONS:
        for _ in range(COST_REPEATS):
            tok = cost_measure(at)
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost_usd(tok).items()}})
    if verbose:
        n = len(rows)
        avgs = {}
        for k in ("input", "output", "cached", "cost_total"):
            avgs[k] = sum(r[k] for r in rows) / n
        min_c = min(r["cost_total"] for r in rows)
        max_c = max(r["cost_total"] for r in rows)
        out_share = 100 * avgs["cost_output"] / avgs["cost_total"] if avgs["cost_total"] else 0
        print("\n" + "=" * 60)
        print(f"COST / query       : ${avgs['cost_total']:.6f}   (Rs {avgs['cost_total'] * USD_TO_INR:.4f})")
        print(f"   min/max         : ${min_c:.6f} / ${max_c:.6f}")
        print(f"   output share    : {out_share:.0f}%")
        daily = avgs["cost_total"] * QUERIES_PER_DAY
        monthly = daily * 30
        print(f"projection @ {QUERIES_PER_DAY}/day : monthly Rs {monthly * USD_TO_INR:.2f}")
    avg = sum(r["cost_total"] for r in rows) / len(rows)
    out_avg = sum(r["cost_output"] for r in rows) / len(rows)
    out_share = 100 * out_avg / avg if avg else 0
    return {
        "cost_per_query_usd": avg, "cost_per_query_inr": avg * USD_TO_INR,
        "avg_input_tokens": sum(r["input"] for r in rows) / len(rows),
        "avg_output_tokens": sum(r["output"] for r in rows) / len(rows),
        "avg_cached_tokens": sum(r["cached"] for r in rows) / len(rows),
        "output_cost_share_pct": out_share,
        "monthly_usd": avg * QUERIES_PER_DAY * 30,
        "budget_pass": avg <= COST_BUDGET_PER_QUERY,
    }


# ============================================================
# RELIABILITY
# ============================================================
REL_REPEATS = 5
MAX_RETRIES = 2
BACKOFF = 0.5

class Reliability:
    def __init__(self):
        self.calls, self.successes, self.failures, self.retries = 0, 0, 0, 0

def call_with_retries(fn, rel):
    rel.calls += 1
    for attempt in range(MAX_RETRIES + 1):
        try:
            fn()
            rel.successes += 1
            return
        except Exception as e:
            if attempt < MAX_RETRIES:
                rel.retries += 1
                time.sleep(BACKOFF * (2 ** attempt))
            else:
                rel.failures += 1
                print(f"[reliability] FAILED after {MAX_RETRIES} retries: {e}")


def run_reliability(at: AgentTarget, verbose=True):
    rel = Reliability()
    print("[reliability] measuring...")
    for q in QUESTIONS:
        for _ in range(REL_REPEATS):
            call_with_retries(lambda: at.invoke(q), rel)
    if verbose:
        calls = rel.calls or 1
        print("\n" + "=" * 60)
        print("RELIABILITY")
        print("=" * 60)
        print(f"total requests : {rel.calls}")
        print(f"successful     : {rel.successes}")
        print(f"failed         : {rel.failures}")
        print(f"success rate   : {100 * rel.successes / calls:.2f}%")
        print(f"error rate      : {100 * rel.failures / calls:.2f}%")
        print(f"retry rate      : {100 * rel.retries / calls:.2f}%")
    calls = rel.calls or 1
    return {
        "total_requests": rel.calls,
        "success_rate": 100 * rel.successes / calls,
        "error_rate": 100 * rel.failures / calls,
        "retry_rate": 100 * rel.retries / calls,
    }


# ============================================================
# ENTRYPOINT
# ============================================================
def run_ops(at: AgentTarget = None, verbose=True):
    at = at or AgentTarget()
    latency = run_latency(at, verbose=verbose)
    cost = run_cost(at, verbose=verbose)
    reliability = run_reliability(at, verbose=verbose)
    snapshot = {}
    snapshot.update({f"latency.{k}": v for k, v in latency.items()})
    snapshot.update({f"cost.{k}": v for k, v in cost.items()})
    snapshot.update({f"reliability.{k}": v for k, v in reliability.items()})
    return snapshot
