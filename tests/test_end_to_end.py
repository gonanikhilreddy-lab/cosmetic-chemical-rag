import unittest

from src.graph.workflow import ask


class EndToEndTests(unittest.TestCase):
    def test_discontinued_filter_is_applied_before_answer(self):
        result = ask("Show products discontinued in 2024 with Titanium dioxide", use_local_model=False)
        self.assertEqual(result["query_plan"]["date_from"], "2024-01-01")
        self.assertEqual(result["summary"]["counts"]["ingredient_records"], 0)
        self.assertIn("This describes the dataset only", result["answer"])

    def test_discontinued_answer_lists_products_with_grouped_citations(self):
        result = ask("Show products discontinued in 2020", use_local_model=False)
        answer = result["answer"]
        self.assertIn("### Matching products (4 total; showing 4)", answer)
        self.assertIn("1. **Lotion Hand Soap**", answer)
        self.assertIn("2. **Pink Lotion Skin Cleanser 820**", answer)
        self.assertIn("**Evidence:** CDPHId 8640 · ChemicalId 13481", answer)
        self.assertIn("**Evidence:** CDPHId 8649 · ChemicalId 13488", answer)
        self.assertNotIn("chemical_name", result["query_plan"]["filters"])
        self.assertEqual(result["query_plan"]["filters"]["date_field"], "discontinued")
        self.assertEqual(result["query_plan"]["filters"]["date_from"], "2020-01-01")
        self.assertGreaterEqual(len(result["step_metrics"]), 7)

    def test_local_model_cannot_omit_sql_list_products(self):
        result = ask("Show products discontinued in 2020", use_local_model=True)
        self.assertEqual(result["summary"]["counts"]["product_count"], 4)
        self.assertEqual(result["model_usage"]["status"], "skipped_sql_grounded_answer")
        for name in ("Lotion Hand Soap", "Pink Lotion Skin Cleanser 820", "Dermacare Pink Lotion Soap", "Ultra Clear Spot Complex"):
            self.assertIn(name, result["answer"])

    def test_reporting_trend_is_aggregated_for_company(self):
        result = ask("Summarize reporting trends for New Avon LLC", use_local_model=False)
        self.assertEqual(result["query_plan"]["entities"]["company"], "New Avon LLC")
        self.assertGreater(len(result["summary"]["aggregate"]["trend"]), 0)
        self.assertGreater(len(result["evidence"]), 0)

    def test_follow_up_inherits_company_and_chemical_then_applies_new_date(self):
        first = ask("Which products from New Avon LLC contain Titanium dioxide?", use_local_model=False)
        second = ask(
            "What about products discontinued in 2010?",
            use_local_model=False,
            conversation_context={"query_plan": first["query_plan"]},
        )
        self.assertEqual(second["query_plan"]["entities"]["company"], "New Avon LLC")
        self.assertEqual(second["query_plan"]["entities"]["chemical"], "Titanium dioxide")
        self.assertEqual(second["query_plan"]["date_from"], "2010-01-01")
        self.assertEqual(second["query_plan"]["inherited_entities"]["company"], "New Avon LLC")
        self.assertGreater(second["summary"]["counts"]["product_count"], 0)
        self.assertTrue(all("CDPHId" in row and "ChemicalId" in row for row in second["evidence"]))

    def test_out_of_scope_question_is_rejected(self):
        result = ask("What is 25 times 30?", use_local_model=False)
        self.assertIn("California cosmetic chemical disclosures", result["answer"])
        self.assertEqual(result["confidence"], "low")

    def test_avon_2016_zero_result_explains_observed_date_coverage(self):
        result = ask("Which AVON brand products were discontinued in 2016?", use_local_model=False)
        self.assertEqual(result["query_plan"]["entities"]["brand"].casefold(), "avon")
        self.assertEqual(result["summary"]["counts"]["product_count"], 0)
        self.assertIn("2009-10-01 through 2015-08-05", result["answer"])
        self.assertIn("268 products", result["answer"])

    def test_brand_comparison_keeps_cas_subcategory_and_date_as_shared_filters(self):
        question = (
            'Compare brands AVON and MARK for CAS 75-07-0 in SubCategory '
            '"Lip Color - Lipsticks, Liners, and Pencils", discontinued in 2010. '
            'Show counts and cited examples for each brand.'
        )
        result = ask(question, use_local_model=False)
        self.assertEqual(result["query_plan"]["comparisons"]["brand"], ["AVON", "MARK"])
        self.assertNotIn("product_name", result["query_plan"]["filters"])
        self.assertEqual(result["query_plan"]["filters"]["cas_number"], "75-07-0")
        self.assertEqual(result["query_plan"]["filters"]["subcategory"], "Lip Color - Lipsticks, Liners, and Pencils")
        comparisons = result["summary"]["aggregate"]["comparison"]
        self.assertEqual([item["entity"] for item in comparisons], ["AVON", "MARK"])
        self.assertTrue(all(item["product_count"] == 0 for item in comparisons))
        self.assertIn("AVON: 0 products", result["answer"])
        self.assertIn("MARK: 0 products", result["answer"])
        self.assertNotIn("Lip:", result["answer"])

    def test_compare_response_cites_examples_per_brand_when_present(self):
        result = ask(
            'Compare brands AVON and MARK for CAS 13463-67-7 in SubCategory '
            '"Lip Color - Lipsticks, Liners, and Pencils", discontinued in 2010.',
            use_local_model=False,
        )
        comparisons = result["summary"]["aggregate"]["comparison"]
        self.assertEqual([item["entity"] for item in comparisons], ["AVON", "MARK"])
        self.assertEqual([item["product_count"] for item in comparisons], [9, 2])
        self.assertIn("**Evidence:** CDPHId", result["answer"])
        for item in comparisons:
            self.assertTrue(all("CDPHId" in row and "ChemicalId" in row for row in item["evidence"]))

    def test_natural_brand_comparison_resolves_lipstick_subcategory(self):
        result = ask(
            "Compare AVON and MARK for CAS 13463-67-7 in the lipstick subcategory, discontinued in 2010.",
            use_local_model=False,
        )
        plan = result["query_plan"]
        self.assertEqual(plan["comparisons"]["brand"], ["AVON", "MARK"])
        self.assertEqual(plan["filters"]["subcategory"], "Lip Color - Lipsticks, Liners, and Pencils")
        self.assertNotIn("product_name", plan["filters"])
        groups = result["summary"]["aggregate"]["comparison"]
        self.assertEqual([group["product_count"] for group in groups], [9, 2])
        self.assertTrue(all(group["evidence"] for group in groups))
        self.assertIn("**Evidence:** CDPHId", result["answer"])

    def test_safety_question_disclaims_health_inferences(self):
        result = ask("Is Titanium dioxide safe in cosmetics?", use_local_model=False)
        self.assertTrue(any("does not determine" in warning for warning in result["warnings"]))

    def test_execution_trace_has_latency_and_work_for_each_step(self):
        result = ask("Which products contain CAS 75-07-0?", limit=1, use_local_model=False)
        names = [step["step"] for step in result["step_metrics"]]
        self.assertEqual(names[0], "input_guardrail")
        self.assertIn("planner", names)
        self.assertIn("structured_retrieval", names)
        self.assertIn("evidence_builder", names)
        self.assertTrue(all(step["latency_ms"] >= 0 and step["details"] for step in result["step_metrics"]))
        usage = result["model_usage"]
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["api_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()