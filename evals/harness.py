"""Shared helpers for deterministic and DeepEval evaluation suites."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "evals" / "goldens"


def load_goldens(path):
    p = Path(path)
    if not p.exists():
        p = GOLDEN_DIR / p.name
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result):
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []
    buckets = {}
    for tr in test_results:
        for metric in getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []:
            name = getattr(metric, "name", "unknown")
            bucket = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            bucket["total"] += 1
            score = getattr(metric, "score", None)
            if score is not None:
                bucket["scores"].append(score)
            if getattr(metric, "success", False):
                bucket["passed"] += 1
    out = {}
    for name, bucket in buckets.items():
        scores = bucket["scores"]
        out[name] = {
            "n": bucket["total"],
            "pass_rate": 100 * bucket["passed"] / bucket["total"] if bucket["total"] else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return out


def print_summary(title, summary):
    print("\n" + "=" * 60 + f"\n{title} (per-metric summary)\n" + "=" * 60)
    for name, stats in summary.items():
        avg = stats["avg_score"]
        print(f"  {name:<26} pass_rate={stats['pass_rate']:5.0f}% avg={avg:.2f} n={stats['n']}" if avg == avg
              else f"  {name:<26} pass_rate={stats['pass_rate']:5.0f}% avg=nan n={stats['n']}")
    print("=" * 60)


def flatten_summary(summary, namespace=""):
    out = {}
    for metric, stats in summary.items():
        slug = metric.strip().lower().replace(" ", "_")
        for stat, value in stats.items():
            key = f"{slug}.{stat}"
            out[f"{namespace}.{key}" if namespace else key] = value
    return out


def select_goldens(goldens, *, limit=None, ids=None, seed=0, stratify_key=None):
    items = list(goldens)
    if ids is not None:
        wanted = list(ids)
        by_id = {str(item.get("id")): item for item in items}
        unknown = [value for value in wanted if str(value) not in by_id]
        if unknown:
            raise ValueError(f"Unknown golden id(s): {', '.join(map(str, unknown))}")
        return [by_id[str(value)] for value in wanted]
    if limit is None or limit >= len(items):
        return items
    if limit <= 0:
        return []
    rng = random.Random(seed)
    if stratify_key is None:
        chosen = {id(item) for item in rng.sample(items, limit)}
    else:
        key_fn = stratify_key if callable(stratify_key) else lambda item: item.get(stratify_key)
        groups = {}
        for item in items:
            groups.setdefault(key_fn(item), []).append(item)
        for group in groups.values():
            rng.shuffle(group)
        chosen_items = []
        keys = sorted(groups, key=lambda x: str(x))
        while len(chosen_items) < limit and any(groups.values()):
            for key in keys:
                if groups[key] and len(chosen_items) < limit:
                    chosen_items.append(groups[key].pop())
        chosen = {id(item) for item in chosen_items}
    return [item for item in items if id(item) in chosen]


def run_evaluate(test_cases, metrics, **kwargs):
    from deepeval import evaluate
    try:
        from deepeval.evaluate import AsyncConfig
        return evaluate(test_cases=test_cases, metrics=metrics,
                        async_config=AsyncConfig(max_concurrent=int(__import__("os").getenv("EVAL_JUDGE_CONCURRENCY", "2"))),
                        **kwargs)
    except (ImportError, TypeError):
        return evaluate(test_cases=test_cases, metrics=metrics, run_async=False, **kwargs)
