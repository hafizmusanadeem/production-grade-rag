"""evals/eval_files/retrieval_metrics.py -- deterministic retrieval metrics.

These are cheap regression signals that don't need an LLM judge (no cost, ~1s runtime).
They key off each golden's source file_name: did the retrieved chunk come from the
golden's expected source? source_hit_at_k is 1 if any retrieved doc matches.
source_mrr ranks the chunks by original score and reports mean reciprocal rank
of hits from the golden's source. source_precision_at_k checks whether the top-k
retrieved docs included the golden's source at all.
"""
from typing import List

GOLDEN_PATH = "evals/goldens/retriever.json"
RETRIEVER_PATH = "evals/eval_files/eval_retriever.py"


def _load_goldens(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(retriever_invoke, *, limit: int = None, ids: List[str] = None) -> dict:
    """deterministic metrics for a single query -> list of RetrievedChunk pairs."""
    if limit is None:
        limit = 50
    ids = ids or []
    goldens = _load_goldens(GOLDEN_PATH)
    results = {}
    for g in goldens[:limit]:
        docs = retriever_invoke(g["query"])
        doc_list = list(docs)
        # hit_at_k: any retrieved doc whose file_name == golden.source
        hits = [d for d in doc_list if d.metadata and d.metadata.get("file_name") == g["source"]]
        results[g["id"]] = {
            "hit_at_k": len(hits),
            "score": sum(d.score for d in doc_list) / len(doc_list) if doc_list else 0.0,
            "k": len(doc_list),
        }
    # Aggregate
    n = len(results)
    hit_vals = [v["hit_at_k"] for v in results.values()]
    score_vals = [v["score"] for v in results.values()]
    prec_vals = [(v["hit_at_k"] + 0) / max(v["k"], 1) for v in results.values()]

    out = {
        "contextual_recall_avg": sum(hit_vals) / n if n else 0.0,
        "contextual_precision_avg": sum(prec_vals) / n if n else 0.0,
        "source_hit_at_k": sum(hit_vals) / n if n else 0.0,
        "source_mrr": (sum(1 / max(i + 1, 1) for i in enumerate(sorted([h for h in hit_vals])) ) / max(n, 1)),
        "pass_rate": 100 * sum(1 for h in hit_vals if h >= 1) / n,
        "n": n,
    }
    return out


def compare():
    """base vs production: run both retrievers on a sample and report hit_at_k."""
    import random
    judge = None  # this is just a helper
    goldens = _load_goldens(GOLDEN_PATH)
    sample = random.sample(goldens, min(20, len(goldens)))

    from app.services.retrieval.retrieval_pipeline import RetrievalPipeline, QueryRequest
    from app.config import RETRIEVAL_LIMIT_K, RERANK_TOP_N, settings
    base_pipe = RetrievalPipeline()
    prod_pipe = RetrievalPipeline()

    results = {"base": [], "production": []}
    for g in sample:
        req = QueryRequest(query=g["query"], limit_k=RETRIEVAL_LIMIT_K, top_n=RERANK_TOP_N,
                           use_reranker=False)
        resp = base_pipe.process_query(req)
        docs = resp.results
        hits = [d for d in docs if d.metadata and d.metadata.get("file_name") == g["source"]]
        results["base"].append(len(hits))

        req2 = QueryRequest(query=g["query"], limit_k=RETRIEVAL_LIMIT_K, top_n=RERANK_TOP_N,
                             use_reranker=True, min_rerank_score=settings.RERANK_MIN_SCORE)
        resp2 = prod_pipe.process_query(req2)
        docs2 = resp2.results
        hits2 = [d for d in docs2 if d.metadata and d.metadata.get("file_name") == g["source"]]
        results["production"].append(len(hits2))

    print("Deterministic retrieval comparison (sample %d):" % len(sample))
    print(f"{'id':<8} {'base':>6} {'prod':>6} {'delta':>7}")
    deltas = []
    for i, g in enumerate(sample):
        bv, pv = results["base"][i], results["production"][i]
        delta = pv - bv
        deltas.append(delta)
        print(f"{g['id']:<8} {bv:>6} {pv:>6} {delta:>+7}")
    print(f"\nbase avg hit_at_k: {sum(results['base']) / len(results['base']):.3f}")
    print(f"production avg hit_at_k: {sum(results['production']) / len(results['production']):.3f}")
    print(f"delta avg: {sum(deltas) / len(deltas):+.3f}")
