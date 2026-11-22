"""Tests for hybrid BM25/FAISS retrieval and reciprocal rank fusion."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize


class FakeEmbeddingService:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts], dtype=np.float32)

    def encode_query(self, _query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeIndexStore:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.root_dir = Path(".")

    def build(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings

    def save(self) -> None:
        return None

    def search(self, query: np.ndarray, *, top_k: int = 5) -> list[dict[str, Any]]:
        scores = self.embeddings @ query
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self.chunks[int(i)], "score": float(scores[int(i)]), "faiss_id": int(i + 1)}
            for i in order
        ]


def chunks() -> list[dict[str, Any]]:
    return [
        {"text": "alpha retrieval engine", "metadata": {"chunk_id": "a"}},
        {"text": "beta vector database", "metadata": {"chunk_id": "b"}},
        {"text": "alpha and beta hybrid search", "metadata": {"chunk_id": "c"}},
    ]


def test_tokenize_supports_unicode_and_casefolding() -> None:
    assert tokenize("ÄPFEL Москва RAG-Search") == ["äpfel", "москва", "rag", "search"]


def test_rrf_combines_rankings_and_deduplicates() -> None:
    a = {"text": "a", "metadata": {"chunk_id": "a"}}
    b = {"text": "b", "metadata": {"chunk_id": "b"}}
    c = {"text": "c", "metadata": {"chunk_id": "c"}}
    fused = reciprocal_rank_fusion(
        [
            [{"chunk": a, "score": 3.0, "faiss_id": 1}, {"chunk": b, "score": 2.0, "faiss_id": 2}],
            [{"chunk": b, "score": 0.9, "faiss_id": 2}, {"chunk": c, "score": 0.8, "faiss_id": 3}],
        ],
        rrf_k=10,
    )

    assert [item["chunk"]["text"] for item in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == pytest.approx(1 / 12 + 1 / 11)
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["vector_rank"] == 1


def test_rrf_validates_parameters() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], rrf_k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[]], weights=[1.0, 2.0])


def test_hybrid_retriever_returns_ranked_chunks_and_component_scores() -> None:
    retriever = HybridRetriever(
        top_k=3,
        candidate_k=3,
        embedding_service=FakeEmbeddingService(),
        index_store=FakeIndexStore(),
    )
    retriever.initialize(chunks(), persist=False)

    results = retriever.retrieve("alpha")

    assert results
    assert all("score" in result for result in results)
    assert all("chunk" in result for result in results)
    assert all("bm25_rank" in result and "vector_rank" in result for result in results)
    assert results[0]["chunk"]["metadata"]["chunk_id"] in {"a", "c"}


def test_empty_query_and_uninitialized_retriever_are_rejected() -> None:
    retriever = HybridRetriever(
        embedding_service=FakeEmbeddingService(),
        index_store=FakeIndexStore(),
    )
    with pytest.raises(ValueError, match="not initialized"):
        retriever.retrieve("alpha")
    retriever.initialize(chunks(), persist=False)
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve("  ")


def test_retrieve_honors_per_request_top_k() -> None:
    retriever = HybridRetriever(
        top_k=3,
        candidate_k=3,
        embedding_service=FakeEmbeddingService(),
        index_store=FakeIndexStore(),
    )
    retriever.initialize(chunks(), persist=False)

    assert len(retriever.retrieve("alpha", top_k=1)) == 1
