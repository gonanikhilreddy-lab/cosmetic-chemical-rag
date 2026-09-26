import unittest

from src.agents.entity_extractor import EntityExtractionAgent
from src.agents.planner import PlannerAgent


class AgentTests(unittest.TestCase):
    def test_planner_extracts_discontinued_year(self):
        plan = PlannerAgent().plan("Show products discontinued in 2024")
        self.assertEqual(plan.date_field, "discontinued")
        self.assertEqual(plan.date_from, "2024-01-01")
        self.assertEqual(plan.date_to, "2025-01-01")
        self.assertEqual(plan.date_operator, "range")

    def test_planner_preserves_comparative_date_operators(self):
        after = PlannerAgent().plan("products reported after January 1, 2020")
        before = PlannerAgent().plan("products reported before January 1, 2020")
        during = PlannerAgent().plan("products reported during 2020")
        between = PlannerAgent().plan("products reported between January 1, 2020 and December 31, 2020")
        self.assertEqual((after.date_operator, after.date_from, after.date_to), ("after", "2020-01-01", None))
        self.assertEqual((before.date_operator, before.date_from, before.date_to), ("before", "2020-01-01", None))
        self.assertEqual((during.date_operator, during.date_from, during.date_to), ("range", "2020-01-01", "2021-01-01"))
        self.assertEqual((between.date_operator, between.date_from, between.date_to), ("between", "2020-01-01", "2020-12-31"))

    def test_entity_extractor_resolves_exact_cas(self):
        question = "Which products contain CAS 75-07-0?"
        plan, hint, clarification = EntityExtractionAgent().extract(question, PlannerAgent().plan(question))
        self.assertEqual(plan.entities["cas"], "75-07-0")
        self.assertIsNone(hint)
        self.assertFalse(clarification)

    def test_company_name_is_not_misread_as_its_brand(self):
        question = "Summarize reporting trends for New Avon LLC"
        plan, _hint, clarification = EntityExtractionAgent().extract(question, PlannerAgent().plan(question))
        self.assertEqual(plan.entities["company"], "New Avon LLC")
        self.assertNotIn("brand", plan.entities)
        self.assertFalse(clarification)

    def test_brand_comparison_keeps_other_constraints_as_shared_filters(self):
        question = (
            'Compare brands AVON and MARK for CAS 75-07-0 in SubCategory '
            '"Lip Color - Lipsticks, Liners, and Pencils", discontinued in 2010.'
        )
        plan, hint, clarification = EntityExtractionAgent().extract(question, PlannerAgent().plan(question))
        self.assertEqual(plan.comparisons["brand"], ["AVON", "MARK"])
        self.assertEqual(plan.entities["cas"], "75-07-0")
        self.assertEqual(plan.entities["subcategory"], "Lip Color - Lipsticks, Liners, and Pencils")
        self.assertNotIn("product", plan.entities)
        self.assertIsNone(hint)
        self.assertFalse(clarification)


if __name__ == "__main__":
    unittest.main()