"""Versioned, atomic persistence for FAISS indexes and their chunk payloads."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray


class IndexStoreError(ValueError):
    """Raised when an index cannot be built, loaded, or searched safely."""


class LocalFaissIndexStore:
    """Persist FAISS vectors and the exact ID-to-chunk map as one snapshot."""

    SCHEMA_VERSION = 1
    _SNAPSHOT_RE = re.compile(r"^[0-9a-f]{32}$")

    def __init__(self, root_dir: str | Path, *, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.root_dir = Path(root_dir)
        self.model_name = model_name
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index: Any | None = None
        self._chunks_by_id: dict[int, dict[str, Any]] = {}
        self.dimension: int | None = None

    @staticmethod
    def _validate_embeddings(
        embeddings: NDArray[np.floating[Any]], expected_rows: int
    ) -> NDArray[np.float32]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows or vectors.shape[1] == 0:
            raise IndexStoreError("Embedding matrix shape does not match the chunk count")
        if not np.isfinite(vectors).all():
            raise IndexStoreError("Embedding matrix contains non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise IndexStoreError("Embedding matrix contains zero-length vectors")
        vectors = vectors / norms[:, np.newaxis]
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def build(
        self,
        chunks: list[dict[str, Any]],
        embeddings: NDArray[np.floating[Any]],
    ) -> None:
        """Build a cosine-similarity index with explicit stable FAISS IDs."""
        if not chunks:
            raise IndexStoreError("Cannot build an index without chunks")
        vectors = self._validate_embeddings(embeddings, len(chunks))
        ids = np.arange(1, len(chunks) + 1, dtype=np.int64)

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(vectors.shape[1])))
        index.add_with_ids(vectors, ids)

        self.index = index
        self.dimension = int(vectors.shape[1])
        self._chunks_by_id = {}
        for faiss_id, chunk in zip(ids.tolist(), chunks, strict=True):
            payload = json.loads(json.dumps(chunk, ensure_ascii=False, default=str))
            payload["faiss_id"] = int(faiss_id)
            metadata = payload.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["faiss_id"] = int(faiss_id)
            self._chunks_by_id[int(faiss_id)] = payload

    @property
    def is_initialized(self) -> bool:
        return self.index is not None and bool(self._chunks_by_id)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def save(self) -> Path:
        """Write an immutable snapshot, then atomically switch CURRENT to it."""
        if not self.is_initialized or self.index is None or self.dimension is None:
            raise IndexStoreError("Index has not been built or loaded")

        snapshot_id = uuid.uuid4().hex
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".snapshot-", dir=self.root_dir)
        )
        final_dir = self.root_dir / snapshot_id
        try:
            index_path = temporary_dir / "index.faiss"
            chunks_path = temporary_dir / "chunks.json"
            manifest_path = temporary_dir / "manifest.json"

            faiss.write_index(self.index, str(index_path))
            chunks_payload = [
                {"faiss_id": faiss_id, "chunk": chunk}
                for faiss_id, chunk in sorted(self._chunks_by_id.items())
            ]
            chunks_path.write_text(
                json.dumps(chunks_payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "count": len(self._chunks_by_id),
                "index_sha256": self._sha256(index_path),
                "chunks_sha256": self._sha256(chunks_path),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            os.replace(temporary_dir, final_dir)
            pointer_tmp = self.root_dir / f".CURRENT-{snapshot_id}"
            pointer_tmp.write_text(snapshot_id, encoding="ascii")
            os.replace(pointer_tmp, self.root_dir / "CURRENT")
            return final_dir
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    @classmethod
    def load(
        cls,
        root_dir: str | Path,
        *,
        expected_model_name: str | None = None,
    ) -> "LocalFaissIndexStore":
        """Restore the current snapshot and verify its checksums and dimensions."""
        root = Path(root_dir)
        pointer = root / "CURRENT"
        if not pointer.is_file():
            raise FileNotFoundError("No persisted FAISS index snapshot was found")

        snapshot_id = pointer.read_text(encoding="ascii").strip()
        if not cls._SNAPSHOT_RE.fullmatch(snapshot_id):
            raise IndexStoreError("Invalid index snapshot pointer")

        snapshot_dir = root / snapshot_id
        manifest_path = snapshot_dir / "manifest.json"
        index_path = snapshot_dir / "index.faiss"
        chunks_path = snapshot_dir / "chunks.json"
        if not all(path.is_file() for path in (manifest_path, index_path, chunks_path)):
            raise IndexStoreError("Index snapshot is incomplete")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise IndexStoreError("Unsupported index snapshot schema")
        model_name = manifest.get("model_name")
        if expected_model_name is not None and model_name != expected_model_name:
            raise IndexStoreError("Persisted index uses a different embedding model")
        if cls._sha256(index_path) != manifest.get("index_sha256"):
            raise IndexStoreError("FAISS index checksum mismatch")
        if cls._sha256(chunks_path) != manifest.get("chunks_sha256"):
            raise IndexStoreError("Chunk payload checksum mismatch")

        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks_by_id = {int(item["faiss_id"]): item["chunk"] for item in payload}
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise IndexStoreError("Unable to deserialize index snapshot") from exc

        expected_count = int(manifest.get("count", -1))
        dimension = int(manifest.get("dimension", -1))
        if (
            expected_count <= 0
            or len(chunks_by_id) != expected_count
            or int(index.ntotal) != expected_count
            or int(index.d) != dimension
        ):
            raise IndexStoreError("Index and chunk metadata are inconsistent")

        store = cls(root, model_name=str(model_name))
        store.index = index
        store._chunks_by_id = chunks_by_id
        store.dimension = dimension
        return store

    def search(
        self,
        query_embedding: NDArray[np.floating[Any]],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search FAISS and resolve the returned IDs to their exact chunk payloads."""
        if not self.is_initialized or self.index is None or self.dimension is None:
            raise IndexStoreError("Index has not been built or loaded")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape != (1, self.dimension):
            raise IndexStoreError("Query embedding has an incompatible dimension")
        if not np.isfinite(query).all():
            raise IndexStoreError("Query embedding contains non-finite values")
        norm = float(np.linalg.norm(query[0]))
        if norm == 0:
            raise IndexStoreError("Query embedding has zero length")
        query = np.ascontiguousarray(query / norm, dtype=np.float32)

        scores, ids = self.index.search(query, min(top_k, len(self._chunks_by_id)))
        results: list[dict[str, Any]] = []
        for score, faiss_id in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if faiss_id < 0:
                continue
            chunk = self._chunks_by_id.get(int(faiss_id))
            if chunk is None:
                raise IndexStoreError(f"FAISS returned an unknown vector ID: {faiss_id}")
            results.append(
                {"chunk": chunk, "score": float(score), "faiss_id": int(faiss_id)}
            )
        return results
