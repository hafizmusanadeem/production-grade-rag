import evals.helper.eval_env

from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from app.services.retrieval.retrieval_pipeline import RetrievalPipeline, QueryRequest
from app.config import settings, RETRIEVAL_LIMIT_K, RERANK_TOP_N
from evals.helper.deepeval_judge import OpenRouterJudge
from evals.eval_files.eval_harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

current_dir = Path(__file__).resolve().parent.parent
GOLDEN_PATH = current_dir / "goldens" / "retriever.json"
JUDGE_MODEL = "openai/gpt-oss-20b:free"   
THRESHOLD = 0.7

def get_judge() -> OpenRouterJudge:
    """Built lazily so importing this module doesn't need an API key."""
    return OpenRouterJudge()

class PipelineRetriever:
    """Adapter: exposes RetrievalPipeline as .invoke(query) -> docs with .page_content."""
    def __init__(self, limit_k: int = RETRIEVAL_LIMIT_K, top_n: int = RERANK_TOP_N):
        self.pipeline = RetrievalPipeline()
        self.use_reranker = use_reranker
        self.limit_k, self.top_n = limit_k, top_n

    def invoke(self, query: str):
        req = QueryRequest(query = query, limit_k = self.limit_k, top_n = self.top_n, use_reranker=self.use_reranker)
        return self.pipeline.process_query(req).results

def run(retriever: PipelineRetriever, judge: OpenRouterJudge | None = None):
    judge = judge or get_judge()
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
            "retriever": retriever.label,
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "limit_k": retriever.limit_k,
            "top_k": retriever.top_n,
            "judge_model": judge.get_model_name(),
            "golden_set": str(GOLDEN_PATH),
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(PipelineRetriever(use_reranker=use_reranker))


if __name__ == "__main__":
    print_summary("retriever", run_local(use_reranker=True))
