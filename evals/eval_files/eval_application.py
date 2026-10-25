"""evals/eval_files/eval_application.py -- application-level quality eval.

Three GEval metrics on LIVE pipeline output: Correctness (truth), Completeness
(coverage). Style is dropped -- the app's responder ("Senior Kubernetes Technical
Architect") has no teaching voice, so Style scores a CampusX lecture and would
mask real regression.
"""
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
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

GOLDEN_PATH = GOLDEN_DIR / "correctness.json"
NAMESPACE = "application"
THRESHOLD = 0.7


def run(agent_target: AgentTarget | None = None, *, limit=None, ids=None, verbose=True):
    if agent_target is None:
        agent_target = AgentTarget()

    goldens = select_goldens(load_goldens(GOLDEN_PATH), limit=limit, ids=ids, seed=0, stratify_key="source")
    judge = get_judge()

    test_cases = []
    errors = 0
    for g in goldens:
        result = agent_target.invoke(g["question"])
        if result.error:
            errors += 1
            continue
        test_cases.append(LLMTestCase(
            input=g["question"],
            actual_output=result.answer,
            expected_output=g["ideal_answer"],
        ))

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false.",
            "A factually accurate answer must score at least 0.9 even if it is shorter.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a factual error."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD, model=judge, strict_mode=False,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points in the expected output.",
            "Check how many key points are addressed in the actual output.",
            "Penalize omissions; judge coverage only.",
            "Do NOT penalize incorrect statements -- correctness is separate.",
            "Extra correct info must NOT lower coverage score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers main points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD, model=judge, strict_mode=False,
    )

    result = run_evaluate(test_cases=test_cases, metrics=[correctness, completeness]) if test_cases else []
    summary = summarize_by_metric(result)
    flat = flatten_summary(summary)
    flat["error_rate"] = errors / len(goldens) if goldens else 0.0
    if verbose:
        print_summary("application", summary)
    return flat


if __name__ == "__main__":
    run()
