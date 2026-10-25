"""evals/eval_files/eval_generator.py -- component eval for the GENERATOR in isolation.

Faithfulness + AnswerRelevancy scored against GOLDEN context (not the retriever's),
so a low score is purely the generator's fault.
"""
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from app.generator import format_context, generate
from evals.harness import (
    flatten_summary,
    load_goldens,
    print_summary,
    run_evaluate,
    select_goldens,
    summarize_by_metric,
)
from evals.helper.deepeval_judge import get_judge

load_dotenv(override=True)

# Golden context file: hand-picked ideal chunks from DATA/true_data per question.
GOLDEN_PATH = "evals/goldens/faithfulness.json"
NAMESPACE = "generator"
THRESHOLD = 0.7


def run(target=None, *, limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(GOLDEN_PATH), limit=limit, ids=ids, seed=0)
    judge = get_judge()

    test_cases = []
    generation_errors = 0
    for g in goldens:
        context = g["ideal_context"]
        if isinstance(context, str):
            context = [context]
        context = format_context(context)
        try:
            answer = generate(g["query"], context, intent="technical")
        except Exception:
            generation_errors += 1
            continue
        if not answer:
            generation_errors += 1
            continue
        test_cases.append(LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context,
        ))

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = run_evaluate(test_cases=test_cases, metrics=metrics) if test_cases else []
    summary = summarize_by_metric(result)
    flat = flatten_summary(summary)
    flat["generation_error_rate"] = generation_errors / len(goldens) if goldens else 0.0
    if verbose:
        print_summary("generator", summary)
    return flat


if __name__ == "__main__":
    run()
