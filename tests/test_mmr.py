import unittest

from src.retrieval.mmr import maximal_marginal_relevance


class MMRTests(unittest.TestCase):
    def test_mmr_returns_ranked_candidates_with_query_relevance(self):
        candidates = [
            {"chemical_name": "A", "rrf_score": 0.03},
            {"chemical_name": "B", "rrf_score": 0.02},
            {"chemical_name": "C", "rrf_score": 0.01},
        ]
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        result = maximal_marginal_relevance(candidates, query, vectors, limit=2, lambda_mult=0.55)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chemical_name"], "A")
        self.assertAlmostEqual(result[0]["vector_score"], 1.0)
        self.assertEqual([item["mmr_rank"] for item in result], [1, 2])

    def test_mmr_rejects_vector_length_mismatch(self):
        with self.assertRaises(ValueError):
            maximal_marginal_relevance([{"chemical_name": "A"}], [1.0], [], limit=1)


if __name__ == "__main__":
    unittest.main()