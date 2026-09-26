import unittest

from src.retrieval.evaluation import retrieval_metrics
from src.retrieval.reranker import rerank_candidates


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_for_rank_one_relevant_result(self):
        metrics = retrieval_metrics(["a", "b", "c"], {"a"}, 3)
        self.assertAlmostEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr@3"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@3"], 1.0)
        self.assertAlmostEqual(metrics["accuracy@1"], 1.0)

    def test_metrics_report_miss_as_zero(self):
        metrics = retrieval_metrics(["x", "y"], {"a"}, 2)
        self.assertEqual(metrics["recall@2"], 0.0)
        self.assertEqual(metrics["mrr@2"], 0.0)
        self.assertEqual(metrics["ndcg@2"], 0.0)

    def test_reranker_rejects_score_equal_to_threshold(self):
        accepted, rejected = rerank_candidates([
            {"chemical_name": "above", "vector_score": 0.7001},
            {"chemical_name": "boundary", "vector_score": 0.70},
            {"chemical_name": "below", "vector_score": 0.69},
        ])
        self.assertEqual([item["chemical_name"] for item in accepted], ["above"])
        self.assertEqual({item["chemical_name"] for item in rejected}, {"boundary", "below"})


if __name__ == "__main__":
    unittest.main()