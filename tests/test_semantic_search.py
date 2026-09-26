import json
import unittest
import urllib.error
import urllib.request

from src.config.settings import OLLAMA_BASE_URL
from src.retrieval.vector_store import build_chemical_index
from src.tools.semantic_search import search_chemicals


def local_embedding_model_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        return any(item["name"].split(":")[0] == "nomic-embed-text" for item in models)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


@unittest.skipUnless(local_embedding_model_available(), "Ollama with nomic-embed-text is not available")
class SemanticSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_chemical_index()

    def test_vector_search_ranks_titanium_dioxide(self):
        matches = search_chemicals("titanium oxide", limit=5)
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0]["chemical_name"], "Titanium dioxide")
        self.assertGreater(float(matches[0]["score"]), 0.7)
        self.assertIn("bm25_rank", matches[0])
        self.assertIn("vector_rank", matches[0])


if __name__ == "__main__":
    unittest.main()