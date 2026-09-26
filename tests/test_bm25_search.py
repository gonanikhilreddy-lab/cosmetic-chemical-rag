import unittest

from src.retrieval.bm25 import search_chemical_bm25


class BM25SearchTests(unittest.TestCase):
    def test_exact_chemical_query_ranks_canonical_name_first(self):
        matches = search_chemical_bm25("Titanium dioxide", limit=5)
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0]["chemical_name"], "Titanium dioxide")
        self.assertGreater(matches[0]["bm25_score"], 0)

    def test_oxide_query_returns_lexical_candidates(self):
        matches = search_chemical_bm25("titanium oxide", limit=5)
        names = {item["chemical_name"] for item in matches}
        self.assertIn("Titanium dioxide", names)
        self.assertTrue(all("bm25_rank" in item for item in matches))


if __name__ == "__main__":
    unittest.main()