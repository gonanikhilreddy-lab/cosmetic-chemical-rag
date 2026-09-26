from __future__ import annotations

import math
from typing import Any, Sequence


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def maximal_marginal_relevance(
    candidates: list[dict[str, Any]],
    query_vector: Sequence[float],
    document_vectors: list[Sequence[float]],
    *,
    limit: int,
    lambda_mult: float = 0.55,
) -> list[dict[str, Any]]:
    """Select relevant but diverse documents with cosine-similarity MMR."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    if len(candidates) != len(document_vectors):
        raise ValueError("Each candidate must have one document vector")

    relevance = [_cosine(query_vector, vector) for vector in document_vectors]
    remaining = set(range(len(candidates)))
    selected: list[int] = []
    while remaining and len(selected) < limit:
        best_index = max(
            remaining,
            key=lambda index: lambda_mult * relevance[index]
            - (1 - lambda_mult) * max(
                (_cosine(document_vectors[index], document_vectors[chosen]) for chosen in selected),
                default=0.0,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)

    return [
        {
            **candidates[index],
            "vector_score": round(relevance[index], 4),
            "mmr_rank": rank,
            "mmr_lambda": lambda_mult,
        }
        for rank, index in enumerate(selected, start=1)
    ]