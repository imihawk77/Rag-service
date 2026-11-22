"""Unit tests for E5 query/document embedding prefixes and validation."""

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingError, EmbeddingService


class FakeEncoder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        self.inputs = texts
        return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)


def test_document_and_query_prefixes_are_distinct_and_normalized() -> None:
    model = FakeEncoder()
    service = EmbeddingService(model=model)

    documents = service.encode_documents(["policy text"])
    query = service.encode_query("where is the policy?")

    assert model.inputs == ["query: where is the policy?"]
    assert np.allclose(documents, [[0.6, 0.8]])
    assert np.allclose(query, [0.6, 0.8])


def test_empty_batch_does_not_load_model() -> None:
    service = EmbeddingService(model_loader=lambda _name: pytest.fail("model should not load"))

    assert service.encode_documents([]).shape == (0, 0)


def test_blank_text_is_rejected() -> None:
    service = EmbeddingService(model=FakeEncoder())

    with pytest.raises(EmbeddingError, match="non-empty"):
        service.encode_queries(["  "])


def test_invalid_model_output_is_rejected() -> None:
    class InvalidEncoder:
        def encode(self, _texts: list[str], **_kwargs: object) -> np.ndarray:
            return np.asarray([[float("nan"), 1.0]], dtype=np.float32)

    with pytest.raises(EmbeddingError, match="non-finite"):
        EmbeddingService(model=InvalidEncoder()).encode_documents(["text"])
