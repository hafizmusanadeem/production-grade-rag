import inspect
from types import SimpleNamespace

from evals.compare import compare, classify
from evals.metric_registry import rule_for
from evals.eval_files import retrieval_metrics
from evals.eval_files import eval_rag_pipeline, eval_application

print("\n===== 1. COMPARE DECISION TESTS =====")

cases = [
    (
        "PASS: unchanged safety metric",
        {"safety.toxicity.avg_toxicity": 0.10},
        {"safety.toxicity.avg_toxicity": 0.10},
        "PASS",
    ),
    (
        "FAIL: safety gate regression",
        {"safety.toxicity.avg_toxicity": 0.10},
        {"safety.toxicity.avg_toxicity": 0.50},
        "FAIL",
    ),
    (
        "REVIEW: quality guardrail regression",
        {"application.answer_quality.avg_score": 0.90},
        {"application.answer_quality.avg_score": 0.70},
        "REVIEW",
    ),
]

for name, baseline_metrics, candidate_metrics, expected in cases:
    verdict, rows = compare(
        {"metrics": baseline_metrics},
        {"metrics": candidate_metrics},
    )
    result = "PASS" if verdict == expected else "FAIL"
    print(f"{name}: expected={expected}, actual={verdict}, test={result}")

print("\n===== 2. MISSING METRIC BEHAVIOUR =====")
for name, baseline_metrics, candidate_metrics in [
    ("candidate metric missing", {"ops.test_pass": True}, {}),
    ("new candidate metric", {}, {"ops.test_pass": True}),
]:
    verdict, rows = compare(
        {"metrics": baseline_metrics},
        {"metrics": candidate_metrics},
    )
    print(name, "=>", verdict)
    print([(r["id"], r["status"]) for r in rows])

print("\n===== 3. RETRIEVAL METRIC TEST =====")
docs = [
    SimpleNamespace(
        metadata={"file_name": "wrong.pdf"},
        score=0.9,
    ),
    SimpleNamespace(
        metadata={"file_name": "expected.pdf"},
        score=0.8,
    ),
]

retrieval_metrics._load_goldens = lambda path: [
    {"id": "synthetic-1", "query": "test", "source": "expected.pdf"}
]

result = retrieval_metrics.run(lambda query: docs, limit=1)
print("Synthetic retrieval result:", result)
print("Expected: hit at rank 2; source MRR should be 0.5")
print("Actual source_mrr:", result.get("source_mrr"))

print("\n===== 4. EVALUATION FUNCTION SIGNATURES =====")
for module in [eval_rag_pipeline, eval_application]:
    print(f"\n{module.__name__}")
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name == "run":
            print("run signature:", inspect.signature(obj))
            print(inspect.getsource(obj))

print("\n===== DIAGNOSTICS FINISHED =====")
