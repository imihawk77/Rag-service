"""Unit tests for retrieval/reranking/generation orchestration."""
from typing import Any
from app.core.rag_chain import RAGChain


class FakeRetriever:
    def __init__(self) -> None:
        self.requested_top_k = None

    def retrieve(self, _question: str, *, top_k: int) -> list[dict[str, Any]]:
        self.requested_top_k = top_k
        return [{
            "chunk": {"text": "The answer is 42.",
                      "metadata": {"source": "guide.pdf", "page": 3}},
            "score": 0.04, "bm25_rank": 2, "vector_rank": 1,
        }]


class FakeReranker:
    def rerank(self, _query: str, results: list[dict[str, Any]], *, top_k: int):
        return results[:top_k]


class FakeLLM:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt, self.user_prompt = system_prompt, user_prompt
        return "Ответ: 42 [1]"


def test_rag_chain_runs_retrieval_reranking_and_generation() -> None:
    retriever, llm = FakeRetriever(), FakeLLM()
    chain = RAGChain(retriever, llm, reranker=FakeReranker(),
                     rerank_candidates=8, top_k=2)
    response = chain.ask("Сколько?")
    assert retriever.requested_top_k == 8
    assert response["answer"] == "Ответ: 42 [1]"
    assert response["sources"][0]["source"] == "guide.pdf"
    assert response["sources"][0]["metadata"]["page"] == 3
    assert "[1] Source: guide.pdf, page: 3" in llm.user_prompt
    assert "только факты" in llm.system_prompt


def test_rag_chain_supports_disabled_reranking() -> None:
    response = RAGChain(FakeRetriever(), FakeLLM(), top_k=1).ask("question")
    assert len(response["sources"]) == 1
