"""Toy ranking-metric checks for reranking; not a production benchmark."""

def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    import math

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def test_reranking_can_improve_mrr_and_ndcg_on_controlled_fixture() -> None:
    relevant = {"target"}
    hybrid_order = ["distractor-a", "distractor-b", "target"]
    reranked_order = ["target", "distractor-a", "distractor-b"]

    assert reciprocal_rank(relevant, reranked_order) > reciprocal_rank(relevant, hybrid_order)
    assert ndcg_at_k(relevant, reranked_order, 3) > ndcg_at_k(relevant, hybrid_order, 3)
