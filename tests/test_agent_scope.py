"""Unit tests for the out-of-scope guard, routing, and node failure handling."""
from types import SimpleNamespace

import pytest

from app.agents.graph import route_planner, route_retriever
from app.agents.nodes import planner as planner_module
from app.agents.nodes import retriever as retriever_module
from app.agents.nodes.decline import NO_CONTEXT_MESSAGE, OUT_OF_SCOPE_MESSAGE, decline_node
from app.agents.nodes.planner import parse_planner_decision, planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import INTENT_CONVERSATIONAL, INTENT_OUT_OF_SCOPE, INTENT_TECHNICAL
from app.config import settings
from app.exceptions import LLMServiceError, RerankError, RetrievalError


# ------------------------------------------------------------ planner parsing
@pytest.mark.parametrize(
    "raw, expected_intent",
    [
        ("OUT_OF_SCOPE", INTENT_OUT_OF_SCOPE),
        ("out_of_scope.", INTENT_OUT_OF_SCOPE),
        ("'OUT OF SCOPE'", INTENT_OUT_OF_SCOPE),
        ("CONVERSATIONAL", INTENT_CONVERSATIONAL),
        ("  conversational  ", INTENT_CONVERSATIONAL),
        ("kubernetes service pod traffic routing", INTENT_TECHNICAL),
    ],
)
def test_parse_planner_decision_intents(raw, expected_intent):
    intent, _ = parse_planner_decision(raw, fallback_query="fallback")
    assert intent == expected_intent


def test_parse_planner_decision_cleans_search_operators():
    intent, query = parse_planner_decision('site:k8s.io "cronjob schedule"', "fallback")
    assert intent == INTENT_TECHNICAL
    assert query == "cronjob schedule"


def test_parse_planner_decision_empty_output_falls_back_to_user_message():
    assert parse_planner_decision("", "what is a Job?") == (INTENT_TECHNICAL, "what is a Job?")


# ------------------------------------------------------------ planner node
def _state(user_message="hi", **extra):
    return {
        "messages": [{"role": "user", "content": user_message}],
        "current_query": "",
        "intent": "",
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
        **extra,
    }


class _FakeLLM:
    def __init__(self, content=None, exc=None):
        self.content, self.exc = content, exc

    def invoke(self, prompt):
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=self.content)


def test_planner_flags_off_topic_question(monkeypatch):
    monkeypatch.setattr(planner_module, "llm", _FakeLLM("OUT_OF_SCOPE"))
    result = planner_node(_state("what is the capital of Pakistan?"))
    assert result["intent"] == INTENT_OUT_OF_SCOPE
    assert "Retrieval: Skipped" in result["plan"]


def test_planner_llm_failure_raises_typed_error(monkeypatch):
    monkeypatch.setattr(planner_module, "llm", _FakeLLM(exc=ConnectionError("boom")))
    with pytest.raises(LLMServiceError):
        planner_node(_state("what is a Job?"))


def test_planner_empty_output_is_logged_fallback_not_silent(monkeypatch):
    monkeypatch.setattr(planner_module, "llm", _FakeLLM(""))
    result = planner_node(_state("what is a Job?"))
    assert result["intent"] == INTENT_TECHNICAL
    assert result["current_query"] == "what is a Job?"
    assert "Planner: fallback to raw question" in result["plan"]


# ------------------------------------------------------------ routing
def test_route_planner():
    assert route_planner({"intent": INTENT_OUT_OF_SCOPE}) == "decline"
    assert route_planner({"intent": INTENT_CONVERSATIONAL}) == "responder"
    assert route_planner({"intent": INTENT_TECHNICAL}) == "retriever"


def test_route_retriever():
    assert route_retriever({"documents": ["CONTENT: x"]}) == "responder"
    assert route_retriever({"documents": []}) == "decline"


# ------------------------------------------------------------ decline node
def test_decline_out_of_scope_message():
    result = decline_node(_state(intent=INTENT_OUT_OF_SCOPE, plan=["Intent: Out of scope"]))
    assert result["final_answer"] == OUT_OF_SCOPE_MESSAGE
    assert "Kubernetes" in result["final_answer"]
    assert result["messages"][0]["role"] == "assistant"


def test_decline_no_context_message():
    result = decline_node(_state(intent=INTENT_TECHNICAL, plan=["Intent: Technical"]))
    assert result["final_answer"] == NO_CONTEXT_MESSAGE


# ------------------------------------------------------------ retriever node
def _chunk(score, text="chunk text"):
    return SimpleNamespace(page_content=text, rerank_score=score)


def _tech_state():
    return _state("what is a Job?", intent=INTENT_TECHNICAL, current_query="kubernetes job", plan=["Intent: Technical"])


def test_retriever_declines_when_nothing_clears_threshold(monkeypatch):
    monkeypatch.setattr(retriever_module, "search_qdrant", lambda **_: [object()])
    monkeypatch.setattr(
        retriever_module, "rerank_documents",
        lambda **_: [_chunk(settings.RERANK_MIN_SCORE / 10)],
    )
    result = retrieve_node(_tech_state())
    assert result["documents"] == []


def test_retriever_keeps_only_relevant_chunks(monkeypatch):
    monkeypatch.setattr(retriever_module, "search_qdrant", lambda **_: [object(), object()])
    monkeypatch.setattr(
        retriever_module, "rerank_documents",
        lambda **_: [_chunk(0.9, "good"), _chunk(settings.RERANK_MIN_SCORE / 10, "noise")],
    )
    result = retrieve_node(_tech_state())
    assert result["documents"] == ["CONTENT: good"]


def test_retriever_no_candidates_skips_rerank(monkeypatch):
    monkeypatch.setattr(retriever_module, "search_qdrant", lambda **_: [])

    def _must_not_run(**_):
        raise AssertionError("rerank should not run with zero candidates")

    monkeypatch.setattr(retriever_module, "rerank_documents", _must_not_run)
    assert retrieve_node(_tech_state())["documents"] == []


def test_retriever_wraps_search_failure(monkeypatch):
    def _boom(**_):
        raise TimeoutError("qdrant down")

    monkeypatch.setattr(retriever_module, "search_qdrant", _boom)
    with pytest.raises(RetrievalError):
        retrieve_node(_tech_state())


def test_retriever_wraps_rerank_failure(monkeypatch):
    def _boom(**_):
        raise RuntimeError("model missing")

    monkeypatch.setattr(retriever_module, "search_qdrant", lambda **_: [object()])
    monkeypatch.setattr(retriever_module, "rerank_documents", _boom)
    with pytest.raises(RerankError):
        retrieve_node(_tech_state())
