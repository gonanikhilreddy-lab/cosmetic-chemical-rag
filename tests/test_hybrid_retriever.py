import unittest
from unittest.mock import patch

from src.retrieval.hybrid_retriever import (
    BM25_CANDIDATE_K,
    FUSED_TOP_K,
    VECTOR_CANDIDATE_K,
    search_chemical_candidates,
)


class HybridRetrieverTests(unittest.TestCase):
    def test_default_depth_is_ten_for_both_retrievers(self):
        vector_rows = [
            {"chemical_name": f"chemical {index}", "score": 0.9 - index / 100}
            for index in range(20)
        ]
        bm25_rows = [
            {"chemical_name": f"chemical {index}", "bm25_score": 10 - index, "bm25_rank": index + 1}
            for index in range(10)
        ]
        with (
            patch("src.retrieval.hybrid_retriever.search_chemical_vectors", return_value=vector_rows) as vector_search,
            patch("src.retrieval.hybrid_retriever.search_chemical_bm25", return_value=bm25_rows) as bm25_search,
        ):
            results = search_chemical_candidates("chemical query")

        vector_search.assert_called_once_with("chemical query", limit=VECTOR_CANDIDATE_K)
        bm25_search.assert_called_once_with("chemical query", limit=BM25_CANDIDATE_K)
        self.assertEqual((VECTOR_CANDIDATE_K, BM25_CANDIDATE_K, FUSED_TOP_K), (10, 10, 10))
        self.assertEqual(len(results), FUSED_TOP_K)


if __name__ == "__main__":
    unittest.main()