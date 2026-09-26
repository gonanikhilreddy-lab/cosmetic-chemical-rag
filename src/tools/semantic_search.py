from src.retrieval.hybrid_retriever import search_chemical_candidates


def search_chemicals(query: str, limit: int = 5) -> list[dict[str, object]]:
    return search_chemical_candidates(query, limit=limit)