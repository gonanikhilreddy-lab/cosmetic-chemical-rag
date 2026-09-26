import unittest

from src.agents.planner import PlannerAgent
from src.tools.date_normalization import normalize_date_expression
from src.tools.structured_query import date_predicate


class DateNormalizationTests(unittest.TestCase):
    def test_between_years_is_exclusive_next_year(self):
        normalized = normalize_date_expression(
            "reported between 2010 and 2012", "reported"
        )
        self.assertEqual(
            (normalized.operator, normalized.date_from, normalized.date_to),
            ("range", "2010-01-01", "2013-01-01"),
        )

    def test_from_to_years_is_a_range(self):
        normalized = normalize_date_expression(
            "reported from 2010 to 2012", "reported"
        )
        self.assertEqual(
            (normalized.operator, normalized.date_from, normalized.date_to),
            ("range", "2010-01-01", "2013-01-01"),
        )

    def test_from_through_years_is_a_range(self):
        normalized = normalize_date_expression(
            "reported from 2010 through 2012", "reported"
        )
        self.assertEqual(
            (normalized.operator, normalized.date_from, normalized.date_to),
            ("range", "2010-01-01", "2013-01-01"),
        )

    def test_existing_comparators_remain_distinct(self):
        self.assertEqual(normalize_date_expression("reported after 2019", "reported").date_from, "2020-01-01")
        self.assertEqual(normalize_date_expression("reported before 2019", "reported").date_from, "2019-01-01")
        self.assertEqual(normalize_date_expression("reported from 2019", "reported").date_from, "2019-01-01")
        self.assertEqual(normalize_date_expression("reported until 2019", "reported").date_to, "2020-01-01")

    def test_planner_and_sql_predicate_agree(self):
        question = "How many unique companies from New Avon LLC reported products containing CAS 13463-67-7 between 2010 and 2012?"
        plan = PlannerAgent().plan(question)
        self.assertEqual((plan.date_operator, plan.date_from, plan.date_to), ("range", "2010-01-01", "2013-01-01"))
        self.assertEqual(
            date_predicate(plan.date_field, plan.date_from, plan.date_to, plan.date_operator),
            "MostRecentDateReported IS NOT NULL AND MostRecentDateReported >= DATE '2010-01-01' AND MostRecentDateReported < DATE '2013-01-01'",
        )


if __name__ == "__main__":
    unittest.main()
