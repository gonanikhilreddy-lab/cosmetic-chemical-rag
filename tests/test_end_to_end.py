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

    def test_standalone_question_does_not_inherit_previous_date(self):
        first = ask("How many products containing titanium dioxide were reported in 2019?", use_local_model=False)
        second = ask(
            "How many unique companies have reported CAS number 13463-67-7?",
            use_local_model=False,
            conversation_context={"query_plan": first["query_plan"]},
        )
        self.assertEqual(second["query_plan"]["intent"], "company_count")
        self.assertIsNone(second["query_plan"]["date_from"])
        self.assertIsNone(second["query_plan"]["date_to"])
        self.assertFalse(second["query_plan"]["inherited_date_constraint"])
        self.assertNotIn("date_field", second["query_plan"]["filters"])

    def test_explicit_follow_up_inherits_context_and_counts_products(self):
        first = ask("Which products from New Avon LLC contain titanium dioxide?", use_local_model=False)
        second = ask(
            "How many of them were discontinued?",
            use_local_model=False,
            conversation_context={"query_plan": first["query_plan"]},
        )
        self.assertEqual(second["query_plan"]["intent"], "product_count")
        self.assertEqual(second["query_plan"]["entities"]["company"], "New Avon LLC")
        self.assertEqual(second["query_plan"]["entities"]["chemical"], "Titanium dioxide")
        self.assertEqual(second["query_plan"]["filters"]["date_operator"], "exists")
        self.assertEqual(second["result_type"], "products")

    def test_follow_up_company_reference_keeps_chemical_scope(self):
        first = ask("How many products contain acetaldehyde?", use_local_model=False)
        second = ask(
            "How many unique companies reported it?",
            use_local_model=False,
            conversation_context={"query_plan": first["query_plan"]},
        )
        third = ask(
            "Which companies are they?",
            use_local_model=False,
            conversation_context={"query_plan": second["query_plan"]},
        )
        self.assertEqual(third["query_plan"]["context_reference"], "company")
        self.assertEqual(third["query_plan"]["inherited_entities"]["chemical"], "Acetaldehyde")
        self.assertEqual(third["query_plan"]["inherited_entities"]["cas"], "75-07-0")
        self.assertEqual(third["result_type"], "companies")

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

    def test_discontinued_without_range_excludes_active_products(self):
        result = ask("Which titanium dioxide products were discontinued?", use_local_model=False)
        self.assertEqual(result["query_plan"]["date_operator"], "exists")
        self.assertEqual(result["query_plan"]["filters"]["chemical_name"], "Titanium dioxide")
        self.assertEqual(result["query_plan"]["filters"]["date_operator"], "exists")
        self.assertTrue(all(row["DiscontinuedDate"] is not None for row in result["evidence"]))

    def test_makeup_alias_is_applied_with_brand_and_chemical(self):
        result = ask("Show me makeup products from Avon that contain titanium dioxide.", use_local_model=False)
        self.assertEqual(result["query_plan"]["filters"]["brand_name"], "AVON")
        self.assertEqual(result["query_plan"]["filters"]["chemical_name"], "Titanium dioxide")
        self.assertEqual(result["query_plan"]["filters"]["primary_category"], "Makeup Products (non-permanent)")
        self.assertTrue(all(row["PrimaryCategory"].strip() == "Makeup Products (non-permanent)" for row in result["evidence"]))

    def test_acetaldehyde_returns_distinct_company_aggregation(self):
        result = ask("How many products contain acetaldehyde, and which companies report it?", use_local_model=False)
        self.assertEqual(result["query_plan"]["entities"]["chemical"], "Acetaldehyde")
        self.assertEqual(result["query_plan"]["entities"]["cas"], "75-07-0")
        self.assertEqual(result["query_plan"]["output_targets"], ["product_count", "company_list"])
        self.assertEqual(result["result_type"], "multi")
        self.assertGreater(result["summary"]["counts"]["product_count"], 0)
        companies = result["summary"]["aggregate"]["companies"]
        self.assertGreater(result["summary"]["aggregate"]["company_count"], 0)
        self.assertEqual(len({row["CompanyName"] for row in companies}), len(companies))
        self.assertFalse(result["evidence"])
        self.assertIn("### Product count", result["answer"])
        self.assertIn("**Distinct products:** 30", result["answer"])
        self.assertIn("### Company count", result["answer"])
        self.assertIn("**Distinct companies:** 8", result["answer"])
        self.assertIn("### Reporting companies", result["answer"])
        for company in companies:
            self.assertIn(f"**{company['CompanyName']}**", result["answer"])

    def test_cas_company_question_returns_companies_not_products(self):
        result = ask("Which companies reported products containing CAS number 13463-67-7?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "company_lookup")
        self.assertEqual(result["result_type"], "companies")
        self.assertGreater(result["summary"]["aggregate"]["company_count"], 0)
        self.assertFalse(result["evidence"])

    def test_company_count_question_returns_distinct_count(self):
        result = ask("How many companies reported products containing CAS number 13463-67-7?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "company_count")
        self.assertEqual(result["query_plan"]["output_targets"], ["company_count"])
        self.assertEqual(result["result_type"], "companies")
        self.assertGreater(result["summary"]["aggregate"]["company_count"], 0)
        self.assertGreaterEqual(
            result["summary"]["aggregate"]["company_count"],
            len(result["summary"]["aggregate"]["companies"]),
        )
        self.assertFalse(result["evidence"])

    def test_product_count_returns_count_only(self):
        result = ask("How many products contain Titanium dioxide?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "product_count")
        self.assertEqual(result["summary"]["counts"]["product_count"], 32010)
        self.assertNotIn("### Matching products", result["answer"])

    def test_top_10_chemicals_uses_distinct_product_aggregation(self):
        result = ask("What are the top 10 chemicals by number of products?", use_local_model=False)
        plan = result["query_plan"]
        aggregate = result["summary"]["aggregate"]
        self.assertEqual((plan["intent"], plan["group_by"], plan["top_n"]), ("aggregation", "chemical", 10))
        self.assertEqual((aggregate["group_count"], aggregate["returned_count"]), (123, 10))
        self.assertEqual(aggregate["groups"][0]["group_value"], "Titanium dioxide")
        self.assertEqual(aggregate["groups"][0]["product_count"], 32010)
        self.assertIn("COUNT(DISTINCT CDPHId)", plan["aggregation_sql"])
        self.assertIn("ORDER BY product_count DESC", plan["aggregation_sql"])
        self.assertIn("LIMIT ?", plan["aggregation_sql"])
        self.assertIn("32,010 products", result["answer"])

    def test_companies_with_most_chemical_products_are_ranked(self):
        result = ask(
            "Which companies have the most products containing Titanium dioxide?",
            use_local_model=False,
        )
        plan = result["query_plan"]
        aggregate = result["summary"]["aggregate"]
        self.assertEqual((plan["intent"], plan["group_by"], plan["aggregation_limit"]), ("aggregation", "company", 20))
        self.assertEqual((aggregate["group_count"], aggregate["returned_count"]), (456, 20))
        self.assertEqual(aggregate["groups"][0]["group_value"], "American International Industries")
        self.assertEqual(aggregate["groups"][0]["product_count"], 1744)
        self.assertTrue(all(
            aggregate["groups"][index]["product_count"] >= aggregate["groups"][index + 1]["product_count"]
            for index in range(len(aggregate["groups"]) - 1)
        ))

    def test_company_product_counts_are_grouped_and_default_limit_is_visible(self):
        result = ask(
            "Show the number of products containing Titanium dioxide for each company.",
            use_local_model=False,
        )
        aggregate = result["summary"]["aggregate"]
        self.assertEqual(result["result_type"], "aggregation")
        self.assertEqual(aggregate["group_count"], 456)
        self.assertEqual(aggregate["returned_count"], 20)
        self.assertIn("top 20 of 456 companies", result["answer"])
        self.assertFalse(result["evidence"])

    def test_top_5_product_categories_are_ranked(self):
        result = ask("What are the top 5 product categories by number of products?", use_local_model=False)
        plan = result["query_plan"]
        groups = result["summary"]["aggregate"]["groups"]
        self.assertEqual((plan["group_by"], plan["top_n"]), ("category", 5))
        self.assertEqual([row["product_count"] for row in groups], [18096, 6920, 5184, 2425, 1768])
        self.assertEqual(groups[0]["group_value"], "Makeup Products (non-permanent)")

    def test_grouped_aggregation_supports_lowest_product_counts(self):
        result = ask("What are the top 5 chemicals by product count ascending?", use_local_model=False)
        plan = result["query_plan"]
        groups = result["summary"]["aggregate"]["groups"]
        self.assertEqual((plan["group_by"], plan["order_direction"], plan["top_n"]), ("chemical", "asc", 5))
        self.assertIn("ORDER BY product_count ASC", plan["aggregation_sql"])
        self.assertTrue(all(
            groups[index]["product_count"] <= groups[index + 1]["product_count"]
            for index in range(len(groups) - 1)
        ))
        self.assertIn("Showing the lowest 5", result["answer"])

    def test_tio2_uses_semantic_resolution_then_exact_duckdb_verification(self):
        result = ask("Find products containing TiO2.", use_local_model=False)
        plan = result["query_plan"]
        candidate = plan["semantic_candidates"][0]
        self.assertEqual(plan["retrieval_mode"], "hybrid")
        self.assertEqual(plan["entities"]["chemical"], "Titanium dioxide")
        self.assertGreater(candidate["relevance_score"], 0.70)
        self.assertEqual(result["summary"]["counts"]["product_count"], 32010)

    def test_ethyl_aldehyde_is_resolved_only_when_semantic_threshold_passes(self):
        result = ask("Find products containing ethyl aldehyde.", use_local_model=False)
        plan = result["query_plan"]
        if result["confidence"] == "needs_clarification":
            self.assertFalse(result["evidence"])
            self.assertFalse(plan.get("filters"))
        else:
            self.assertEqual(plan["entities"]["chemical"], "Acetaldehyde")
            self.assertGreater(plan["semantic_candidates"][0]["relevance_score"], 0.70)
            self.assertEqual(result["summary"]["counts"]["product_count"], 30)

    def test_dangerous_chemicals_is_not_mapped_to_a_specific_chemical(self):
        result = ask("Find products with dangerous chemicals.", use_local_model=False)
        self.assertEqual(result["confidence"], "needs_clarification")
        self.assertFalse(result["evidence"])
        self.assertNotIn("chemical_name", result["query_plan"].get("filters", {}))

    def test_makeup_titanium_products_discontinued_filter_is_preserved(self):
        result = ask(
            "Find makeup products containing Titanium dioxide that were discontinued.",
            use_local_model=False,
        )
        filters = result["query_plan"]["filters"]
        self.assertEqual(result["summary"]["counts"], {"ingredient_records": 7326, "product_count": 2503})
        self.assertEqual(filters["primary_category"], "Makeup Products (non-permanent)")
        self.assertEqual(filters["date_operator"], "exists")
        self.assertTrue(all(row["DiscontinuedDate"] is not None for row in result["evidence"]))

    def test_multi_output_company_then_product_counts(self):
        result = ask("How many companies report titanium dioxide, and how many products contain it?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "multi_output")
        self.assertEqual(result["query_plan"]["output_targets"], ["company_count", "product_count"])
        self.assertEqual(result["summary"]["aggregate"]["company_count"], 456)
        self.assertEqual(result["summary"]["counts"]["product_count"], 32010)
        self.assertEqual(result["summary"]["counts"]["ingredient_records"], 93480)
        self.assertIn("**Distinct companies:** 456", result["answer"])
        self.assertIn("**Distinct products:** 32,010", result["answer"])
        self.assertIn("**Ingredient records:** 93,480", result["answer"])

    def test_multi_output_product_then_company_counts(self):
        result = ask("How many products contain titanium dioxide, and how many companies report it?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "multi_output")
        self.assertEqual(result["query_plan"]["output_targets"], ["product_count", "company_count"])
        self.assertEqual(result["summary"]["aggregate"]["company_count"], 456)
        self.assertEqual(result["summary"]["counts"]["product_count"], 32010)

    def test_chemical_count_returns_distinct_chemicals(self):
        result = ask("How many chemicals are reported for ANEW EYELIFTING SERUM SHADOW-ALL SHADES?", use_local_model=False)
        self.assertEqual(result["query_plan"]["intent"], "chemical_count")
        self.assertEqual(result["summary"]["aggregate"]["chemical_count"], 2)
        self.assertEqual(result["result_type"], "chemicals")

    def test_complex_company_count_applies_category_and_date_filters(self):
        result = ask("Among makeup products reported after 2019, how many unique companies reported titanium dioxide?", use_local_model=False)
        filters = result["query_plan"]["filters"]
        self.assertEqual(result["query_plan"]["intent"], "company_count")
        self.assertEqual(filters["primary_category"], "Makeup Products (non-permanent)")
        self.assertEqual((filters["date_operator"], filters["date_from"], filters["date_to"]), ("after", "2020-01-01", None))
        self.assertGreater(result["summary"]["aggregate"]["company_count"], 0)

    def test_discontinued_makeup_product_count_applies_all_filters(self):
        result = ask("How many discontinued titanium dioxide makeup products were reported by New Avon LLC?", use_local_model=False)
        filters = result["query_plan"]["filters"]
        self.assertEqual(result["query_plan"]["intent"], "product_count")
        self.assertEqual(filters["company_name"], "New Avon LLC")
        self.assertEqual(filters["chemical_name"], "Titanium dioxide")
        self.assertEqual(filters["primary_category"], "Makeup Products (non-permanent)")
        self.assertEqual(filters["date_operator"], "exists")
        self.assertGreater(result["summary"]["counts"]["product_count"], 0)

    def test_reported_after_uses_strict_lower_bound(self):
        result = ask("Find all products containing titanium dioxide that were reported after January 1, 2020.", use_local_model=False)
        plan = result["query_plan"]
        self.assertEqual((plan["date_operator"], plan["date_from"], plan["date_to"]), ("after", "2020-01-01", None))
        self.assertTrue(all(row["MostRecentDateReported"] > "2020-01-01" for row in result["evidence"]))

    def test_reported_before_uses_strict_upper_bound(self):
        result = ask("Find all products containing titanium dioxide that were reported before January 1, 2020.", use_local_model=False)
        plan = result["query_plan"]
        self.assertEqual((plan["date_operator"], plan["date_from"], plan["date_to"]), ("before", "2020-01-01", None))
        self.assertTrue(all(row["MostRecentDateReported"] < "2020-01-01" for row in result["evidence"]))

    def test_reported_during_year_uses_half_open_range(self):
        result = ask("Find all products containing titanium dioxide that were reported during 2020.", use_local_model=False)
        plan = result["query_plan"]
        self.assertEqual((plan["date_operator"], plan["date_from"], plan["date_to"]), ("range", "2020-01-01", "2021-01-01"))
        self.assertTrue(all("2020-01-01" <= row["MostRecentDateReported"] < "2021-01-01" for row in result["evidence"]))

    def test_reported_between_uses_inclusive_bounds(self):
        result = ask("Find all products containing titanium dioxide that were reported between January 1, 2020 and December 31, 2020.", use_local_model=False)
        plan = result["query_plan"]
        self.assertEqual((plan["date_operator"], plan["date_from"], plan["date_to"]), ("range", "2020-01-01", "2021-01-01"))
        self.assertTrue(all("2020-01-01" <= row["MostRecentDateReported"] <= "2020-12-31" for row in result["evidence"]))

    def test_between_years_plan_and_sql_predicate_use_exclusive_next_year(self):
        result = ask(
            "How many unique companies from New Avon LLC reported products containing CAS 13463-67-7 between 2010 and 2012?",
            use_local_model=False,
        )
        plan = result["query_plan"]
        self.assertEqual((plan["date_from"], plan["date_to"], plan["date_operator"]), ("2010-01-01", "2013-01-01", "range"))
        self.assertEqual(
            plan["sql_date_predicate"],
            "MostRecentDateReported IS NOT NULL AND MostRecentDateReported >= DATE '2010-01-01' AND MostRecentDateReported < DATE '2013-01-01'",
        )
        self.assertEqual(result["summary"]["aggregate"]["company_count"], 1)

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