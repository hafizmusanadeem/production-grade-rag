from types import SimpleNamespace

from evals.eval_files.retrieval_metrics import metrics_from_docs


def d(name): return SimpleNamespace(metadata={"file_name":name},page_content=name,rerank_score=.8)


def test_source_second_has_mrr_half():
    out=metrics_from_docs([{"id":"1","source":"expected.pdf"}],{"1":[d("wrong.pdf"),d("expected.pdf")]})
    assert out["source_mrr"]==.5 and out["source_hit_at_k"]==1


def test_no_docs_is_miss():
    out=metrics_from_docs([{"id":"1","source":"expected.pdf"}],{"1":[]})
    assert out["source_hit_at_k"]==0 and out["source_mrr"]==0


def test_multiple_hits_bounded():
    out=metrics_from_docs([{"id":"1","source":"expected.pdf"}],{"1":[d("expected.pdf"),d("expected.pdf")]})
    assert out["source_hit_at_k"]==1
