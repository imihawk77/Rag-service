"""Optional Cross-Encoder reranking for hybrid retrieval candidates."""
from collections.abc import Callable, Sequence
from typing import Any


class CrossEncoderReranker:
    """Rerank retrieved chunks by query-passage relevance; load the model lazily."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", *,
                 model: Any | None = None, model_loader: Callable[[str], Any] | None = None,
                 batch_size: int = 16) -> None:
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
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: Sequence[dict[str, Any]], *,
               top_k: int | None = None) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not results:
            return []
        pairs: list[list[str]] = []
        for result in results:
            chunk = result.get("chunk", {})
            text = chunk.get("text") if isinstance(chunk, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every reranking result must contain non-empty chunk text")
            pairs.append([query.strip(), text])
        scores = self._get_model().predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        if len(scores) != len(results):
            raise ValueError("Cross-Encoder returned an unexpected number of scores")
        ranked = []
        for result, raw_score in zip(results, scores):
            item = dict(result)
            item["retrieval_score"] = result.get("score")
            item["rerank_score"] = float(raw_score)
            item["score"] = float(raw_score)
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked if top_k is None else ranked[:top_k]
