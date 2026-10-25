"""Deterministic source-hit metrics for retrieval."""
from evals.harness import load_goldens, select_goldens

GOLDEN_PATH = "evals/goldens/retriever.json"


def retrieve_all(retriever, goldens):
    return {str(g["id"]): list(retriever.invoke(g["query"])) for g in goldens}


def metrics_from_docs(goldens, docs_by_id):
    hits, mrrs, precisions, empties = [], [], [], []
    for g in goldens:
        docs = list(docs_by_id.get(str(g["id"]), []))
        files = [(getattr(d, "metadata", {}) or {}).get("file_name") for d in docs]
        source = g.get("source")
        hit = source in files
        hits.append(float(hit))
        mrrs.append(1 / (files.index(source) + 1) if hit else 0.0)
        precisions.append(files.count(source) / len(files) if docs else 0.0)
        empties.append(float(not docs))
    n = len(goldens)
    avg = lambda values: sum(values) / n if n else 0.0
    return {"source_hit_at_k": avg(hits), "source_mrr": avg(mrrs),
            "source_precision_at_k": avg(precisions), "empty_context_rate": avg(empties), "n": n}


def run(retriever, *, limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(GOLDEN_PATH), limit=limit, ids=ids, seed=0, stratify_key="source")
    return metrics_from_docs(goldens, retrieve_all(retriever, goldens))
