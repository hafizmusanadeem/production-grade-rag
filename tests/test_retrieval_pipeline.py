from app.config import RERANK_TOP_N, RETRIEVAL_LIMIT_K
from app.services.retrieval import retrieval_pipeline as rp
from app.services.retrieval.qdrant_service import RetrievedChunk
from app.services.retrieval.retrieval_pipeline import QueryRequest, RetrievalPipeline


def _chunks(n):
    return [RetrievedChunk(chunk_id=str(i), page_content=f"c{i}", score=1 - i / 100) for i in range(n)]


def test_defaults_match_production():
    req = QueryRequest(query="q")
    assert (req.limit_k, req.top_n) == (RETRIEVAL_LIMIT_K, RERANK_TOP_N)
    assert req.use_reranker is True


def test_base_mode_skips_reranker_and_keeps_vector_order(monkeypatch):
    monkeypatch.setattr(rp, "search_qdrant", lambda **_: _chunks(15))

    def _must_not_run(**_):
        raise AssertionError("reranker must not run in base mode")

    monkeypatch.setattr(rp, "rerank_documents", _must_not_run)
    resp = RetrievalPipeline().process_query(QueryRequest(query="q", use_reranker=False))
    assert [r.chunk_id for r in resp.results] == ["0", "1", "2", "3", "4"]
    assert resp.retrieved_count == 15


def test_rerank_mode_calls_reranker(monkeypatch):
    monkeypatch.setattr(rp, "search_qdrant", lambda **_: _chunks(15))
    called = {}

    def _fake_rerank(**kw):
        called["top_n"] = kw["top_n"]
        return []

    monkeypatch.setattr(rp, "rerank_documents", _fake_rerank)
    RetrievalPipeline().process_query(QueryRequest(query="q"))
    assert called["top_n"] == RERANK_TOP_N



def test_threshold_applies_only_to_reranked_results(monkeypatch):
    monkeypatch.setattr(rp,"search_qdrant",lambda **_: _chunks(2))
    def fake(**_):
        return [type("C",(),{"chunk_id":"good","page_content":"good","original_score":.8,"rerank_score":.9,"metadata":{}})(),
                type("C",(),{"chunk_id":"bad","page_content":"bad","original_score":.7,"rerank_score":.1,"metadata":{}})()]
    monkeypatch.setattr(rp,"rerank_documents",fake)
    pipe=RetrievalPipeline()
    assert [x.chunk_id for x in pipe.process_query(QueryRequest(query="q",min_rerank_score=.5)).results]==["good"]
    assert len(pipe.process_query(QueryRequest(query="q",use_reranker=False,min_rerank_score=.99)).results)==2
