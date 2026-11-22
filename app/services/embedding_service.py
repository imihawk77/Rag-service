"""E5 embedding adapter with explicit query/document task prefixes."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EmbeddingError(ValueError):
    """Raised when text cannot be converted to valid embeddings."""


class EmbeddingService:
    """Lazily load Sentence Transformers and expose separate E5 task methods."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        *,
        model: Any | None = None,
        model_loader: Callable[[str], Any] | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model
        self._model_loader = model_loader

    def _get_model(self) -> Any:
        if self._model is None:
            if self._model_loader is not None:
                self._model = self._model_loader(self.model_name)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str], *, prefix: str) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("All texts must be non-empty strings")

        prepared = [f"{prefix}{text.strip()}" for text in texts]
        try:
            vectors = self._get_model().encode(
                prepared,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("Embedding model failed to encode text") from exc

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts) or array.shape[1] == 0:
            raise EmbeddingError("Embedding model returned an invalid vector shape")
        if not np.isfinite(array).all():
            raise EmbeddingError("Embedding model returned non-finite values")

        norms = np.linalg.norm(array, axis=1)
        if np.any(norms == 0):
            raise EmbeddingError("Embedding model returned a zero-length vector")
        array = array / norms[:, np.newaxis]
        return np.ascontiguousarray(array, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode passages with the E5 passage instruction prefix."""
        return self._encode(texts, prefix="passage: ")

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode search queries with the E5 query instruction prefix."""
        return self._encode(texts, prefix="query: ")

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Encode one query and return a single vector."""
        return self.encode_queries([text])[0]
