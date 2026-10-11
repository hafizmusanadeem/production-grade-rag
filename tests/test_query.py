from fastapi.testclient import TestClient

from app.exceptions import LLMServiceError, RetrievalError
from app.main import app


class FakeAgent:
    def invoke(self, state, config=None):
        return {
            "final_answer": "ok",
            "documents": [],
            "intent": "conversational",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }


class RaisingAgent:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, state, config=None):
        raise self.exc


class EmptyAnswerAgent:
    def invoke(self, state, config=None):
        return {"final_answer": "", "documents": [], "plan": []}


# raise_server_exceptions=False so the catch-all 500 handler is exercised
# the way a real client would see it, instead of the exception re-raising in-test.
client = TestClient(app, raise_server_exceptions=False)


def test_query_returns_answer(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", FakeAgent())
    response = client.post("/query", json={"q": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ok"
    assert body["intent"] == "conversational"


def test_blank_question_is_rejected_with_422(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", FakeAgent())
    assert client.post("/query", json={"q": "   "}).status_code == 422
    assert client.post("/query", json={"q": ""}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def test_overlong_question_is_rejected_with_422(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", FakeAgent())
    assert client.post("/query", json={"q": "x" * 2001}).status_code == 422


def test_bad_thread_id_is_rejected_with_422(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", FakeAgent())
    response = client.post("/query", json={"q": "hi", "thread_id": "../etc/passwd"})
    assert response.status_code == 422


def test_retrieval_failure_maps_to_503_with_safe_message(monkeypatch):
    monkeypatch.setattr(
        "app.main.rag_agent",
        RaisingAgent(RetrievalError("qdrant timeout at https://secret-cluster:6333")),
    )
    response = client.post("/query", json={"q": "what is a Job?"})
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "retrieval_unavailable"
    assert error["error_id"]
    assert "secret-cluster" not in response.text  # internals never leak


def test_llm_failure_maps_to_502(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", RaisingAgent(LLMServiceError("provider 429")))
    response = client.post("/query", json={"q": "what is a Job?"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_unavailable"


def test_unexpected_exception_is_500_and_does_not_leak(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", RaisingAgent(RuntimeError("db password is hunter2")))
    response = client.post("/query", json={"q": "what is a Job?"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "hunter2" not in response.text


def test_empty_agent_answer_is_an_error_not_a_blank_200(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", EmptyAnswerAgent())
    response = client.post("/query", json={"q": "hi"})
    assert response.status_code == 500
