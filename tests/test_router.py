import unittest

from src.graph.workflow import ask


class RouterTests(unittest.TestCase):
    def test_exact_cas_routes_to_structured_sql(self):
        result = ask("Which products contain CAS 75-07-0?", limit=2, use_local_model=False)
        self.assertEqual(result["query_plan"]["retrieval_mode"], "structured")
        self.assertEqual(len(result["evidence"]), 2)
        self.assertIn("ChemicalId", result["evidence"][0])

    def test_fuzzy_chemical_routes_through_semantic_then_sql(self):
        result = ask("Which products contain titanium oxide?", limit=2, use_local_model=False)
        self.assertEqual(result["query_plan"]["retrieval_mode"], "hybrid")
        self.assertEqual(result["query_plan"]["entities"]["chemical"], "Titanium dioxide")
        self.assertEqual(len(result["evidence"]), 2)
        candidate = result["query_plan"]["semantic_candidates"][0]
        self.assertIn("vector_rank", candidate)
        self.assertIn("bm25_rank", candidate)

    def test_ambiguous_avon_pauses_for_clarification(self):
        result = ask("Show me Avon", use_local_model=False)
        self.assertEqual(result["confidence"], "needs_clarification")
        self.assertFalse(result["evidence"])

    def test_safety_question_stops_before_retrieval(self):
        result = ask("Is Titanium dioxide safe to use during pregnancy?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "safety_question")
        self.assertFalse(result["evidence"])
        self.assertIn("pregnancy-safety or clinical evidence", result["answer"])
        steps = [step["step"] for step in result["step_metrics"]]
        self.assertEqual(steps, ["input_guardrail", "planner", "safety_question", "output_guardrail"])


if __name__ == "__main__":
    unittest.main()