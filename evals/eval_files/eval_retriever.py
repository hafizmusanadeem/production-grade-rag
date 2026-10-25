"""Retriever eval: base, reranked and production threshold variants."""
import argparse

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from app.config import RERANK_TOP_N, RETRIEVAL_LIMIT_K, settings
from app.services.retrieval.retrieval_pipeline import QueryRequest, RetrievalPipeline
from evals.config import GOLDEN_DIR
from evals.eval_files.retrieval_metrics import metrics_from_docs, retrieve_all
from evals.harness import (
    flatten_summary,
    load_goldens,
    run_evaluate,
    select_goldens,
    summarize_by_metric,
)
from evals.helper.deepeval_judge import get_judge

NAMESPACE = "retriever"
THRESHOLD = 0.7
GOLDEN_PATH = GOLDEN_DIR / "retriever.json"
VARIANTS = {
    "base": {"use_reranker": False, "min_rerank_score": None},
    "reranked": {"use_reranker": True, "min_rerank_score": None},
    "production": {"use_reranker": True, "min_rerank_score": settings.RERANK_MIN_SCORE},
}


class PipelineRetriever:
    def __init__(self, variant="production", *, limit_k=RETRIEVAL_LIMIT_K, top_n=RERANK_TOP_N):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}")
        self.variant = variant
        self.options = VARIANTS[variant]
        self.pipeline = RetrievalPipeline()
        self.limit_k, self.top_n = limit_k, top_n

    @property
    def label(self):
        return self.variant

    def invoke(self, query):
        return self.pipeline.process_query(QueryRequest(
            query=query, limit_k=self.limit_k, top_n=self.top_n,
            use_reranker=self.options["use_reranker"],
            min_rerank_score=self.options["min_rerank_score"],
        )).results


def _evaluate(goldens, variant, judge=None, verbose=True):
    retriever = PipelineRetriever(variant)
    docs_by_id = retrieve_all(retriever, goldens)
    flat = metrics_from_docs(goldens, docs_by_id)
    cases = []
    for g in goldens:
        docs = docs_by_id[str(g["id"])]
        if docs:
            cases.append(LLMTestCase(input=g["query"], expected_output=g["ideal_answer"],
                                     retrieval_context=[d.page_content for d in docs],
                                     actual_output="(generator not evaluated)"))
    if cases:
        judge = judge or get_judge()
        result = run_evaluate(test_cases=cases, metrics=[
            ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
            ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ], hyperparameters={"variant": variant, "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                            "chunk_size": settings.CHUNK_SIZE, "chunk_overlap": settings.CHUNK_OVERLAP,
                            "limit_k": retriever.limit_k, "top_n": retriever.top_n,
                            "min_rerank_score": retriever.options["min_rerank_score"],
                            "judge_model": judge.get_model_name()})
        flat.update(flatten_summary(summarize_by_metric(result)))
    return flat


def run(target=None, *, variant="production", limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(GOLDEN_PATH), limit=limit, ids=ids, seed=0, stratify_key="source")
    result = _evaluate(goldens, variant, verbose=verbose)
    if verbose:
        print(f"Retriever {variant}: {result}")
    return result


def run_local():
    return run()


def run_comparison(limit=20, verbose=True):
    goldens = select_goldens(load_goldens(GOLDEN_PATH), limit=limit, seed=0, stratify_key="source")
    judge = get_judge()
    results = {name: _evaluate(goldens, name, judge=judge, verbose=False)
               for name in ("base", "reranked", "production")}
    if verbose:
        print("Retriever comparison:", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="production")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None
    if args.compare:
        run_comparison(args.limit or 20, verbose=not args.quiet)
    else:
        print(run(variant=args.variant, limit=args.limit, ids=ids, verbose=not args.quiet))


if __name__ == "__main__":
    main()
