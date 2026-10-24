"""evals/eval_files/eval_rag_pipeline.py -- quality eval of the full RAG pipeline.

Runs the triad (ContextualRelevancy + Faithfulness + AnswerRelevancy) on the
faithfulness golden set using LIVE pipeline output (context comes from the agent's
retrieval, answer from the generator).
"""
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric

from evals.target import AgentTarget
from evals.harness import load_goldens, summarize_by_metric, flatten_summary, print_summary

load_dotenv(override=True)

GOLDEN_PATH = "evals/goldens/faithfulness.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def run(agent_target: AgentTarget | None = None):
    if agent_target is None:
        agent_target = AgentTarget()

    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        result = agent_target.invoke(g["query"])
        test_cases.append(LLMTestCase(
            input=g["query"],
            actual_output=result.answer,
            retrieval_context=result.context,     # post-prefix-strip chunk text
        ))

    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    summary = summarize_by_metric(result)
    flat = flatten_summary("pipeline", summary)
    print_summary("rag_pipeline", flat)
    return flat


if __name__ == "__main__":
    run()
