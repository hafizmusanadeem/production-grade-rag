from dotenv import load_dotenv
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from app.services.retrieval.retrieval_pipeline import RetrievalPipeline, QueryRequest
from app.config import settings
from evals.helper.deepeval_judge import OpenRouterJudge
from evals.eval_files.eval_harness import load_goldens, summarize_by_metric, print_summary
from eval.eval_files.eval_retriever import PipelineRetriever, get_judge, run
load_dotenv()

current_dir = Path(__file__).resolve().parent.parent
GOLDEN_PATH = current_dir / "goldens" / "retriever.json"
JUDGE_MODEL = "openai/gpt-oss-20b:free"   
judge = OpenRouterJudge(JUDGE_MODEL)
THRESHOLD = 0.7


class PipelineRetriever:
    """Adapter: exposes RetrievalPipeline as .invoke(query) -> docs with .page_content."""
    def __init__(self, limit_k: int = 10, top_n: int = 5):
        self.pipeline = RetrievalPipeline()
        self.limit_k, self.top_n = limit_k, top_n

    def invoke(self, query: str):
        req = QueryRequest(query = query, limit_k = self.limit_k, top_n = self.top_n)
        return self.pipeline.process_query(req).results

def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",          # vs "reranked" when you swap it in
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": 3,
            "judge_model": judge,
            "golden_set": str(GOLDEN_PATH),
        },
    )
    return summarize_by_metric(result)

def compare(base: dict, reranked: dict) -> None:
    print("\n" + "=" * 78)
    print("BASE vs RERANKED  (avg score / pass rate)")
    print("=" * 78)
    print(f"  {'metric':<24}{'base':>14}{'reranked':>14}{'delta avg':>12}")
    for name in sorted(set(base) | set(reranked)):
        b, r = base.get(name), reranked.get(name)
        if not b or not r:
            print(f"  {name:<24}{'(missing in one run)':>40}")
            continue
        delta = r["avg_score"] - b["avg_score"]
        print(
            f"  {name:<24}"
            f"{b['avg_score']:>8.2f}/{b['pass_rate']:>3.0f}%"
            f"{r['avg_score']:>8.2f}/{r['pass_rate']:>3.0f}%"
            f"{delta:>+12.2f}"
        )
    print("=" * 78)


def run_comparison() -> dict:
    judge = get_judge()  # one judge instance for both runs
    base = run(PipelineRetriever(use_reranker=False), judge)
    reranked = run(PipelineRetriever(use_reranker=True), judge)
    compare(base, reranked)
    return {"base": base, "reranked": reranked}


if __name__ == "__main__":
    run_comparison()
