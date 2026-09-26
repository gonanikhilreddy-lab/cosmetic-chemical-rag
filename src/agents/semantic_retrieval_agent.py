from src.tools.semantic_search import search_chemicals
from rapidfuzz import fuzz, process

from src.retrieval.reranker import FINAL_TOP_K, rerank_candidates
from src.tools.structured_query import entity_values


class SemanticRetrievalAgent:
    def search(self, hint: str) -> tuple[list[dict[str, object]], str | None, bool, list[dict[str, object]]]:
        try:
            candidates = search_chemicals(hint, limit=FINAL_TOP_K)
        except (ConnectionError, FileNotFoundError, OSError, RuntimeError):
            matches = process.extract(hint, entity_values("chemical"), scorer=fuzz.WRatio, limit=5, score_cutoff=45)
            candidates = [
                {"chemical_name": value, "score": round(float(score) / 100, 4), "retrieval": "fuzzy-fallback"}
                for value, score, _index in matches
            ]
        accepted, rejected = rerank_candidates(candidates, limit=FINAL_TOP_K)
        if not accepted:
            return accepted, None, True, rejected
        top_score = accepted[0]["relevance_score"]
        next_score = accepted[1]["relevance_score"] if len(accepted) > 1 else 0.0
        if len(accepted) > 1 and top_score - next_score < 0.025:
            return accepted, None, True, rejected
        return accepted, str(accepted[0]["chemical_name"]), False, rejected