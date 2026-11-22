"""Unit tests for FAISS ID mapping and durable snapshot restoration."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.storage import index_store
from app.storage.index_store import IndexStoreError, LocalFaissIndexStore


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return len(self.ids)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.concatenate([self.ids, ids])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = (query @ self.vectors.T)[0]
        order = np.argsort(scores)[::-1][:k]
        return scores[order].reshape(1, -1), self.ids[order].reshape(1, -1)


@pytest.fixture
def fake_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_index(index: FakeIndex, path: str) -> None:
        with open(path, "wb") as stream:
            pickle.dump(index, stream)

    def read_index(path: str) -> FakeIndex:
        with open(path, "rb") as stream:
            return pickle.load(stream)

    monkeypatch.setattr(
        index_store,
        "faiss",
        SimpleNamespace(
            IndexFlatIP=lambda dimension: FakeIndex(dimension),
            IndexIDMap2=lambda index: index,
            write_index=write_index,
            read_index=read_index,
        ),
    )


def sample_chunks() -> list[dict[str, object]]:
    return [
        {"text": "first passage", "metadata": {"chunk_id": "c1", "page": 1}},
        {"text": "second passage", "metadata": {"chunk_id": "c2", "page": 2}},
    ]


def test_search_maps_faiss_ids_to_the_correct_chunks(tmp_path: Path, fake_faiss: None) -> None:
    store = LocalFaissIndexStore(tmp_path, model_name="test-e5")
    store.build(sample_chunks(), np.asarray([[1, 0], [0, 1]], dtype=np.float32))

    results = store.search(np.asarray([0, 1], dtype=np.float32), top_k=2)

    assert results[0]["chunk"]["text"] == "second passage"
    assert results[0]["faiss_id"] == 2
    assert results[0]["score"] == pytest.approx(1.0)


def test_index_and_chunk_map_survive_restart(tmp_path: Path, fake_faiss: None) -> None:
    store = LocalFaissIndexStore(tmp_path, model_name="test-e5")
    store.build(sample_chunks(), np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    store.save()

    restored = LocalFaissIndexStore.load(tmp_path, expected_model_name="test-e5")
    results = restored.search(np.asarray([1, 0], dtype=np.float32))

    assert results[0]["chunk"]["metadata"]["chunk_id"] == "c1"
    assert restored.dimension == 2


def test_model_mismatch_is_rejected(tmp_path: Path, fake_faiss: None) -> None:
    store = LocalFaissIndexStore(tmp_path, model_name="model-a")
    store.build(sample_chunks(), np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    store.save()

    with pytest.raises(IndexStoreError, match="different embedding model"):
        LocalFaissIndexStore.load(tmp_path, expected_model_name="model-b")


def test_checksum_corruption_is_rejected(tmp_path: Path, fake_faiss: None) -> None:
    store = LocalFaissIndexStore(tmp_path, model_name="test-e5")
    snapshot = store.build(sample_chunks(), np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    store.save()
    current = (tmp_path / "CURRENT").read_text(encoding="ascii").strip()
    chunks_path = tmp_path / current / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")

    with pytest.raises(IndexStoreError, match="checksum"):
        LocalFaissIndexStore.load(tmp_path)


def test_empty_or_invalid_embeddings_are_rejected(tmp_path: Path, fake_faiss: None) -> None:
    store = LocalFaissIndexStore(tmp_path, model_name="test-e5")

    with pytest.raises(IndexStoreError, match="without chunks"):
        store.build([], np.empty((0, 2), dtype=np.float32))
    with pytest.raises(IndexStoreError, match="zero-length"):
        store.build(sample_chunks(), np.asarray([[0, 0], [1, 0]], dtype=np.float32))


def test_query_dimension_is_validated(tmp_path: Path, fake_faiss: None) -> None:
    store = LocalFaissIndexStore(tmp_path, model_name="test-e5")
    store.build(sample_chunks(), np.asarray([[1, 0], [0, 1]], dtype=np.float32))

    with pytest.raises(IndexStoreError, match="incompatible dimension"):
        store.search(np.asarray([1, 0, 0], dtype=np.float32))
