import json
import unittest
import urllib.error
import urllib.request

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.graph.workflow import ask


def local_answer_model_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        return any(item["name"].split(":")[0] == OLLAMA_MODEL.split(":")[0] for item in models)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


@unittest.skipUnless(local_answer_model_available(), "Configured local answer model is not available")
class LocalModelTests(unittest.TestCase):
    def test_local_model_synthesizes_from_retrieved_evidence(self):
        result = ask("Summarize chemicals for Brand AVON", limit=2, use_local_model=True)
        self.assertTrue(result["evidence"])
        self.assertIn("CDPHId", result["answer"])
        self.assertFalse(any("no local" in warning.lower() for warning in result["warnings"]))
        usage = result["model_usage"]
        self.assertEqual(usage["status"], "completed")
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(usage["api_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()