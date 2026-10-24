"""evals/eval_files/eval_application.py -- application-level quality eval.

Three GEval metrics on LIVE pipeline output: Correctness (truth), Completeness
(coverage). Style is dropped -- the app's responder ("Senior Kubernetes Technical
Architect") has no teaching voice, so Style scores a CampusX lecture and would
mask real regression.
"""
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from evals.target import AgentTarget
from evals.harness import load_goldens, summarize_by_metric, flatten_summary, print_summary

load_dotenv(override=True)

GOLDEN_PATH = "evals/goldens/correctness.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def run(agent_target: AgentTarget | None = None):
    if agent_target is None:
        agent_target = AgentTarget()

    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        result = agent_target.invoke(g["question"])
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
        threshold=THRESHOLD, model=JUDGE_MODEL, strict_mode=False,
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
        threshold=THRESHOLD, model=JUDGE_MODEL, strict_mode=False,
    )

    result = evaluate(test_cases=test_cases, metrics=[correctness, completeness])
    summary = summarize_by_metric(result)
    flat = flatten_summary("application", summary)
    print_summary("application", flat)
    return flat


if __name__ == "__main__":
    run()
