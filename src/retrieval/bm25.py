import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.tools.structured_query import entity_values


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


@lru_cache(maxsize=1)
def _chemical_index() -> tuple[tuple[str, ...], BM25Okapi]:
    names = tuple(entity_values("chemical"))
    corpus = [tokenize(name) for name in names]
    return names, BM25Okapi(corpus)


def search_chemical_bm25(query: str, limit: int = 10) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    names, index = _chemical_index()
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ordered = sorted(range(len(names)), key=lambda item: scores[item], reverse=True)
    return [
        {
            "chemical_name": names[item],
            "bm25_score": round(float(scores[item]), 6),
            "bm25_rank": rank,
        }
        for rank, item in enumerate(ordered[:limit], start=1)
        if scores[item] > 0
    ]