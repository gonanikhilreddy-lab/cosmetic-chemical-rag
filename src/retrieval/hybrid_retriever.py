from src.retrieval.bm25 import search_chemical_bm25
from src.retrieval.vector_store import search_chemical_vectors


RRF_K = 60
VECTOR_CANDIDATE_K = 10
BM25_CANDIDATE_K = 10
FUSED_TOP_K = 10


def search_chemical_candidates(query: str, limit: int = FUSED_TOP_K) -> list[dict[str, object]]:
    vector_matches: list[dict[str, object]] = []
    bm25_matches = search_chemical_bm25(query, limit=BM25_CANDIDATE_K)
    vector_error = None
    try:
        vector_matches = search_chemical_vectors(query, limit=VECTOR_CANDIDATE_K)
    except Exception as error:
        vector_error = type(error).__name__

    merged: dict[str, dict[str, object]] = {}
    for rank, item in enumerate(vector_matches, start=1):
        name = str(item["chemical_name"])
        candidate = merged.setdefault(name, {"chemical_name": name, "vector_score": 0.0, "bm25_score": 0.0})
        candidate["vector_score"] = float(item["score"])
        candidate["vector_rank"] = rank
        candidate["rrf_score"] = float(candidate.get("rrf_score", 0.0)) + 1 / (RRF_K + rank)

    for item in bm25_matches:
        name = str(item["chemical_name"])
        candidate = merged.setdefault(name, {"chemical_name": name, "vector_score": 0.0, "bm25_score": 0.0})
        candidate["bm25_score"] = float(item["bm25_score"])
        candidate["bm25_rank"] = int(item["bm25_rank"])
        candidate["rrf_score"] = float(candidate.get("rrf_score", 0.0)) + 1 / (RRF_K + int(item["bm25_rank"]))

    ranked = sorted(merged.values(), key=lambda item: float(item.get("rrf_score", 0.0)), reverse=True)
    for item in ranked:
        item["score"] = round(float(item.get("vector_score", 0.0)), 4)
        item["retrieval"] = "vector+bm25" if item.get("vector_rank") and item.get("bm25_rank") else (
            "vector" if item.get("vector_rank") else "bm25"
        )
        if vector_error:
            item["vector_error"] = vector_error
        item["rrf_score"] = round(float(item.get("rrf_score", 0.0)), 6)
    return ranked[:limit]