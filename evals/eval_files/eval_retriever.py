"""evals/eval_files/eval_retriever.py -- component eval for the RETRIEVER.

Measures retrieval quality via DeepEval's ContextualRecall / ContextualPrecision
metrics on the golden set, using the REAL RetrievalPipeline (production path).

Variants: base (vector order, no reranker), reranked (rerank, no threshold),
production (rerank + threshold = what serves traffic).
"""
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from app.services.retrieval.retrieval_pipeline import RetrievalPipeline, QueryRequest
from app.config import settings, RETRIEVAL_LIMIT_K, RERANK_TOP_N
from app.services.retrieval.qdrant_service import RetrievedChunk
from evals.helper.deepeval_judge import OpenRouterJudge
from evals.harness import load_goldens, summarize_by_metric, flatten_summary, print_summary

load_dotenv(override=True)

GOLDEN_PATH = "evals/goldens/retriever.json"
JUDGE_MODEL = "openai/gpt-oss-20b:free"
THRESHOLD = 0.7

# Default variant mirrors production config; run with --variant to switch.
DEFAULT_VARIANT = "production"


def get_judge():
    return OpenRouterJudge(JUDGE_MODEL)


class PipelineRetriever:
    """Adapter: exposes RetrievalPipeline as .invoke(query) -> docs with .page_content."""

    def __init__(self, *, limit_k: int = RETRIEVAL_LIMIT_K, top_n: int = RERANK_TOP_N,
                 use_reranker: bool = True, min_rerank_score: float | None = None):
        self.pipeline = RetrievalPipeline()
        self.use_reranker = use_reranker
        self.min_rerank_score = min_rerank_score if min_rerank_score is not None else settings.RERANK_MIN_SCORE
        self.limit_k, self.top_n = limit_k, top_n

    @property
    def label(self):
        mode = "reranked" if self.use_reranker else "base"
        if self.min_rerank_score is not None and self.min_rerank_score != settings.RERANK_MIN_SCORE:
            mode += f"+min{self.min_rerank_score:.3f}"
        return mode

    def invoke(self, query: str):
        req = QueryRequest(query=query, limit_k=self.limit_k, top_n=self.top_n,
                           use_reranker=self.use_reranker,
                           min_rerank_score=self.min_rerank_score)
        resp = self.pipeline.process_query(req)
        return resp.results


def _run_variant(variant: str, judge: OpenRouterJudge | None = None):
    judge = judge or get_judge()
    pipe = PipelineRetriever(use_reranker=True if variant in ("base", "reranked", "production") else False)
    if variant == "base":
        pipe = PipelineRetriever(use_reranker=False)
    elif variant == "reranked":
        pass  # default already uses reranker
    elif variant == "production":
        pipe = PipelineRetriever(use_reranker=True,
                                 min_rerank_score=settings.RERANK_MIN_SCORE)
    else:
        raise ValueError(f"unknown variant {variant}")

    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        docs = pipe.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in docs]

        test_cases.append(LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        ))

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": pipe.label,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "limit_k": pipe.limit_k,
            "top_k": pipe.top_n,
            "min_rerank_score": pipe.min_rerank_score,
            "judge_model": judge.get_model_name(),
            "golden_set": GOLDEN_PATH,
        },
    )
    summary = summarize_by_metric(result)
    return flatten_summary("retriever", summary)


def run(retriever: PipelineRetriever, judge: OpenRouterJudge | None = None):
    judge = judge or get_judge()
    return _run_variant(retriever.label, judge)


def run_local():
    return run(PipelineRetriever())


def run_comparison(limit: int = 20, verbose: bool = True):
    """Compare base vs production on a stratified sample of goldens."""
    import random
    judge = get_judge()
    full = load_goldens(GOLDEN_PATH)
    sample = random.sample(full, min(limit, len(full)))

    base_flat = flatten_summary("retriever", _run_variant("base", judge))
    prod_flat = flatten_summary("retriever", _run_variant("production", judge))

    print("\n" + "=" * 78)
    print("RETRIEVER COMPARISON  (sample size: %d)" % len(sample))
    print("=" * 78)
    print(f"{'metric':<44} {'base':>10} {'prod':>10} {'delta':>10}")
    print("-" * 78)
    all_mids = sorted(set(base_flat) | set(prod_flat))
    for mid in all_mids:
        bv = base_flat.get(mid, "")
        pv = prod_flat.get(mid, "")
        if isinstance(bv, float) and isinstance(pv, float):
            d = pv - bv
            delta_str = f"{d:+.4g}"
        else:
            delta_str = ""
        print(f"{mid:<44} {str(bv):>10} {str(pv):>10} {delta_str:>10}")
    print("=" * 78)
    return {"base": base_flat, "production": prod_flat}


if __name__ == "__main__":
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VARIANT
    if variant == "compare":
        run_comparison(limit=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        res = _run_variant(variant, get_judge())
        print(f"retriever.{variant}: {res}")
