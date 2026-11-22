"""Deterministic retrieval metric helpers; real corpus cases are added by users."""

import json
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("retrieval_dataset.json")


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def test_retrieval_dataset_has_queries_and_relevance_labels() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assert len(dataset) >= 5
    assert all(item["query"].strip() and item["relevant_chunk_ids"] for item in dataset)


def test_recall_and_mrr_metrics() -> None:
    retrieved = ["x", "b", "a"]
    relevant = ["a", "b"]
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank_at_k(retrieved, relevant, 3) == 0.5
