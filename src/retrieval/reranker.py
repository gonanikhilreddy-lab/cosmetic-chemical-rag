from typing import Any


RELEVANCE_THRESHOLD = 0.70
FINAL_TOP_K = 10


def rerank_candidates(
    candidates: list[dict[str, Any]],
    *,
    threshold: float = RELEVANCE_THRESHOLD,
    limit: int = FINAL_TOP_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rerank by query/document cosine relevance and reject scores <= threshold."""
    scored = []
    for candidate in candidates:
        relevance = candidate.get("vector_score")
        if relevance is None and candidate.get("retrieval") == "fuzzy-fallback":
            relevance = candidate.get("score", 0.0)
        item = {**candidate, "relevance_score": round(float(relevance or 0.0), 4)}
        scored.append(item)

    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    accepted = []
    rejected = []
    for item in scored:
        item["accepted"] = item["relevance_score"] > threshold
        item["rerank_reason"] = (
            f"relevance {item['relevance_score']:.4f} > {threshold:.2f}"
            if item["accepted"]
            else f"rejected: relevance {item['relevance_score']:.4f} is not > {threshold:.2f}"
        )
        (accepted if item["accepted"] else rejected).append(item)
    return accepted[:limit], rejected