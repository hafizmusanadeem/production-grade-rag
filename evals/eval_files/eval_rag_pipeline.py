"""evals/eval_files/eval_rag_pipeline.py -- quality eval of the full RAG pipeline.

Runs the triad (ContextualRelevancy + Faithfulness + AnswerRelevancy) on the
faithfulness golden set using LIVE pipeline output (context comes from the agent's
retrieval, answer from the generator).
"""
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from evals.harness import (
    flatten_summary,
    load_goldens,
    print_summary,
    run_evaluate,
    select_goldens,
    summarize_by_metric,
)
from evals.helper.deepeval_judge import get_judge
from evals.target import AgentTarget

load_dotenv(override=True)

from evals.config import GOLDEN_DIR

GOLDEN_PATH = GOLDEN_DIR / "retriever.json"
NAMESPACE = "pipeline"
THRESHOLD = 0.7


def run(agent_target: AgentTarget | None = None, *, limit=None, ids=None, verbose=True):
    if agent_target is None:
        agent_target = AgentTarget()

    goldens = select_goldens(load_goldens(GOLDEN_PATH), limit=limit, ids=ids, seed=0, stratify_key="source")
    judge = get_judge()

    test_cases = []
    declined = errors = 0
    for g in goldens:
        result = agent_target.invoke(g["query"])
        if result.error:
            errors += 1
            continue
        if result.declined:
            declined += 1
            continue
        if not result.context:
            continue
        test_cases.append(LLMTestCase(input=g["query"], expected_output=g.get("ideal_answer", ""),
                                      actual_output=result.answer, retrieval_context=result.context))

    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = run_evaluate(test_cases=test_cases, metrics=metrics) if test_cases else []
    summary = summarize_by_metric(result)
    flat = flatten_summary(summary)
    flat["decline_on_answerable_rate"] = declined / len(goldens) if goldens else 0.0
    flat["error_rate"] = errors / len(goldens) if goldens else 0.0
    if verbose:
        print_summary("rag_pipeline", summary)
    return flat


if __name__ == "__main__":
    run()
