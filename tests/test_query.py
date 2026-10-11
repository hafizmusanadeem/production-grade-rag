from fastapi.testclient import TestClient

from app.main import app


class FakeAgent:
    def invoke(self, state, config=None):
        return {
            "final_answer": "ok",
            "documents": [],
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }


def test_query_returns_answer(monkeypatch):
    monkeypatch.setattr("app.main.rag_agent", FakeAgent())
    client = TestClient(app)
    response = client.post("/query", json={"q": "hi"})
    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
