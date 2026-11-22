"""Hybrid BM25 + FAISS retrieval with Reciprocal Rank Fusion (RRF)."""

import re
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi

from app.services.embedding_service import EmbeddingService
from app.storage.index_store import LocalFaissIndexStore


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode text consistently for BM25 indexing and querying."""
    return _TOKEN_RE.findall(text.casefold())


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists using 1 / (rrf_k + 1-based rank).

    Results are deduplicated by FAISS ID when present, otherwise by chunk_id.
    The first occurrence supplies the chunk payload; component scores/ranks are
    retained in the fused result for diagnostics.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")
    if weights is not None and (
        len(weights) != len(ranked_lists) or any(weight < 0 for weight in weights)
    ):
        raise ValueError("weights must be non-negative and match ranked list count")

    fused: dict[str, dict[str, Any]] = {}
    for list_index, results in enumerate(ranked_lists):
        weight = weights[list_index] if weights is not None else 1.0
        seen_in_list: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk = result.get("chunk", {})
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            identity_value = metadata.get("chunk_id")
            if identity_value is None:
                identity_value = result.get("faiss_id")
            if identity_value is None:
                identity_value = chunk.get("text") if isinstance(chunk, dict) else None
            if identity_value is None:
                raise ValueError("Every retrieval result must have a stable chunk identity")

            identity = str(identity_value)
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)

            if identity not in fused:
                fused[identity] = {
                    "chunk": chunk,
                    "faiss_id": result.get("faiss_id"),
                    "score": 0.0,
                    "bm25_score": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "vector_rank": None,
                }
            entry = fused[identity]
            entry["score"] += weight / (rrf_k + rank)
            if list_index == 0:
                entry["bm25_score"] = result.get("score")
                entry["bm25_rank"] = rank
            elif list_index == 1:
                entry["vector_score"] = result.get("score")
                entry["vector_rank"] = rank

    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)


class HybridRetriever:
    """Combine lexical BM25 and dense FAISS rankings with weighted RRF."""

    def __init__(
        self,
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        weight_bm25: float = 0.3,
        weight_vector: float = 0.7,
        top_k: int = 5,
        *,
        rrf_k: int = 60,
        candidate_k: int = 20,
        embedding_service: Any | None = None,
        index_store: Any | None = None,
        index_dir: str = "./data/indexes",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if candidate_k < top_k:
            candidate_k = top_k
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if weight_bm25 < 0 or weight_vector < 0:
            raise ValueError("retrieval weights must be non-negative")
        if weight_bm25 + weight_vector <= 0:
            raise ValueError("at least one retrieval weight must be positive")

        self.embedding_model_name = embedding_model_name
        self.weight_bm25 = float(weight_bm25)
        self.weight_vector = float(weight_vector)
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.embedding_service = embedding_service or EmbeddingService(
            embedding_model_name
        )
        self.index_store = index_store or LocalFaissIndexStore(
            index_dir, model_name=embedding_model_name
        )
        self.chunks: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.is_initialized = False

    def initialize(
        self,
        chunks: list[dict[str, Any]],
        *,
        persist: bool = True,
    ) -> None:
        """Build lexical and vector indexes from the same ordered chunk collection."""
        if not chunks:
            raise ValueError("No chunks provided for initialization")

        valid_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.get("text"), str) and chunk["text"].strip()
        ]
        if not valid_chunks:
            raise ValueError("No non-empty chunk texts provided for initialization")

        texts = [chunk["text"] for chunk in valid_chunks]
        tokenized = [tokenize(text) for text in texts]
        if not any(tokenized):
            raise ValueError("No searchable tokens found in chunks")

        embeddings = self.embedding_service.encode_documents(texts)
        self.index_store.build(valid_chunks, embeddings)
        if persist:
            self.index_store.save()

        self.chunks = valid_chunks
        self.bm25 = BM25Okapi(tokenized)
        self.is_initialized = True

    def load_persisted(self) -> None:
        """Restore the FAISS snapshot and rebuild BM25 from its exact chunk map."""
        self.index_store = LocalFaissIndexStore.load(
            self.index_store.root_dir,
            expected_model_name=self.embedding_model_name,
        )
        chunks_by_id = self.index_store._chunks_by_id
        self.chunks = [
            payload for _, payload in sorted(chunks_by_id.items(), key=lambda item: item[0])
        ]
        tokenized = [tokenize(str(chunk.get("text", ""))) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized and any(tokenized) else None
        self.is_initialized = self.bm25 is not None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve candidates from both indexes and fuse them with weighted RRF."""
        if not self.is_initialized or self.bm25 is None:
            raise ValueError("Retriever not initialized.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        result_count = self.top_k if top_k is None else top_k
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = tokenize(query)
        if query_tokens:
            lexical_scores = self.bm25.get_scores(query_tokens)
            lexical_order = sorted(
                range(len(self.chunks)),
                key=lambda position: float(lexical_scores[position]),
                reverse=True,
            )
            lexical_results = [
                {
                    "chunk": self.chunks[position],
                    "score": float(lexical_scores[position]),
                    "faiss_id": position + 1,
                }
                for position in lexical_order
                if float(lexical_scores[position]) > 0
            ][: self.candidate_k]
        else:
            lexical_results = []

        query_vector = self.embedding_service.encode_query(query)
        vector_results = self.index_store.search(
            query_vector, top_k=min(self.candidate_k, len(self.chunks))
        )

        fused = reciprocal_rank_fusion(
            [lexical_results, vector_results],
            rrf_k=self.rrf_k,
            weights=[self.weight_bm25, self.weight_vector],
        )
        return fused[:result_count]
