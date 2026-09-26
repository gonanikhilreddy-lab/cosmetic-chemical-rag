from __future__ import annotations

import math
from typing import Sequence


def retrieval_metrics(ranked: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """Compute standard top-K metrics for a ranked list and binary relevance set."""
    if k < 1:
        raise ValueError("k must be positive")
    top = list(ranked[:k])
    hits = [1 if item in relevant else 0 for item in top]
    relevant_count = len(relevant)
    first_relevant = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    precision = sum(hits) / k
    recall = sum(hits) / relevant_count if relevant_count else 0.0
    average_precision = 0.0
    seen = 0
    for index, hit in enumerate(hits, start=1):
        if hit:
            seen += 1
            average_precision += seen / index
    average_precision = average_precision / min(relevant_count, k) if relevant_count else 0.0
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, start=1))
    ideal_hits = min(relevant_count, k)
    ideal_dcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"hit_rate@{k}": 1.0 if sum(hits) else 0.0,
        f"accuracy@1": float(bool(top) and top[0] in relevant),
        f"mrr@{k}": 1 / first_relevant if first_relevant else 0.0,
        f"map@{k}": average_precision,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }