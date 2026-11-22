"""Unit tests for Cross-Encoder reranking without model downloads."""
from typing import Any
import pytest
from app.core.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs: list[list[str]], **_kwargs: Any) -> list[float]:
        return [0.2 if "weak" in pair[1] else 0.9 for pair in pairs]


def test_reranker_orders_results_and_preserves_retrieval_score() -> None:
    results = [
        {"chunk": {"text": "weak passage", "metadata": {"source": "a"}}, "score": 0.08},
        {"chunk": {"text": "strong passage", "metadata": {"source": "b"}}, "score": 0.12},
    ]
    ranked = CrossEncoderReranker(model=FakeCrossEncoder()).rerank("query", results, top_k=1)
    assert len(ranked) == 1
    assert ranked[0]["chunk"]["metadata"]["source"] == "b"
    assert ranked[0]["retrieval_score"] == 0.12
    assert ranked[0]["rerank_score"] == pytest.approx(0.9)


def test_reranker_empty_results_do_not_load_model() -> None:
    reranker = CrossEncoderReranker(model_loader=lambda _name: pytest.fail("must stay lazy"))
    assert reranker.rerank("query", []) == []


def test_reranker_validates_query_and_top_k() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank(" ", [])
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("query", [], top_k=0)
