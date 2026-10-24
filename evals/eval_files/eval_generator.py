"""evals/eval_files/eval_generator.py -- component eval for the GENERATOR in isolation.

Faithfulness + AnswerRelevancy scored against GOLDEN context (not the retriever's),
so a low score is purely the generator's fault.
"""
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from app.generator import generate
from evals.harness import load_goldens, summarize_by_metric, flatten_summary, print_summary

load_dotenv(override=True)

# Golden context file: hand-picked ideal chunks from DATA/true_data per question.
GOLDEN_PATH = "evals/goldens/faithfulness.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def run():
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        context = g["ideal_context"]          # known-good chunk text (no prefix)
        answer = generate(g["query"], context)
        test_cases.append(LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context,
        ))

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    summary = summarize_by_metric(result)
    flat = flatten_summary("generator", summary)
    print_summary("generator", flat)
    return flat


if __name__ == "__main__":
    run()
