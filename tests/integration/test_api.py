"""HTTP-level integration tests for the FastAPI application."""
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, state


class FakeRAGChain:
    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        return {
            "question": question,
            "answer": "Grounded answer [1]",
            "sources": [
                {
                    "text": "The source text",
                    "source": "fixture.pdf",
                    "score": 0.05,
                    "metadata": {"page": 2},
                }
            ][:top_k or 1],
        }


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(state, "retriever", object())
    monkeypatch.setattr(state, "rag_chain", FakeRAGChain())
    monkeypatch.setattr(
        state,
        "chunks",
        [{"text": "The source text", "metadata": {"source": "fixture.pdf"}}],
    )
    with TestClient(app) as test_client:
        yield test_client
    state.retriever = None
    state.rag_chain = None
    state.chunks = []


def test_liveness_endpoint(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint_reports_loaded_index(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["chunks_count"] == 1


def test_query_endpoint_returns_answer_and_sources(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"question": "What does the fixture say?", "top_k": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Grounded answer [1]"
    assert payload["sources"][0]["source"] == "fixture.pdf"
    assert payload["sources"][0]["metadata"]["page"] == 2


def test_query_rejects_blank_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post("/query", json={"question": "valid", "top_k": 0})
    assert response.status_code == 422
