"""Regression tests for real bugs found during the first live eval run.

The agent state is checkpointed by LangGraph's MemorySaver, which msgpack-
serialises every value. numpy scalars (FlashRank returns float32 scores) are
NOT msgpack serialisable, so a raw score in ``sources`` crashed every
AgentTarget.invoke with a TypeError.
"""
import numpy as np
import pytest


class _Chunk:
    def __init__(self, rerank_score, original_score):
        self.rerank_score = rerank_score
        self.original_score = original_score
        self.chunk_id = "c1"
        self.page_content = "text"
        self.metadata = {"file_name": "a.docx"}


def test_retriever_sources_are_plain_floats():
    from app.agents.nodes.retriever import _build_sources
    chunks = [_Chunk(np.float32(0.42), np.float32(0.11))]
    sources = _build_sources(chunks)
    assert sources[0]["rerank_score"] == pytest.approx(0.42)
    assert type(sources[0]["rerank_score"]) is float
    assert type(sources[0]["original_score"]) is float


def test_retriever_sources_msgpack_serialisable():
    import ormsgpack

    from app.agents.nodes.retriever import _build_sources
    chunks = [_Chunk(np.float32(0.42), np.float32(0.11))]
    packed = ormsgpack.packb({"sources": _build_sources(chunks)})
    assert ormsgpack.unpackb(packed)["sources"][0]["rerank_score"] == pytest.approx(0.42)
