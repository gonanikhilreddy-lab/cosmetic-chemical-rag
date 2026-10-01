import unittest
from datetime import date

from src.tools import structured_query as queries


class StructuredQueryTests(unittest.TestCase):
    def test_cas_search_preserves_source_identifiers(self):
        results = queries.find_by_cas("75-07-0")
        self.assertGreater(len(results), 0)
        self.assertTrue((results["CasNumber"].str.strip() == "75-07-0").all())
        self.assertIn("CDPHId", results.columns)
        self.assertIn("ChemicalId", results.columns)

    def test_combined_exact_filters(self):
        results = queries.search_cosmetics(
            brand_name="AVON",
            chemical_name="Titanium dioxide",
            limit=5,
        )
        self.assertGreater(len(results), 0)
        self.assertTrue((results["BrandName"].str.strip() == "AVON").all())
        self.assertTrue((results["ChemicalName"].str.strip() == "Titanium dioxide").all())

    def test_category_match_trims_dataset_whitespace(self):
        results = queries.find_by_category(
            "Hair Care Products (non-coloring)",
            "Hair Shampoos (making a cosmetic claim)",
        )
        self.assertGreater(len(results), 0)
        self.assertTrue((results["PrimaryCategory"].str.strip() == "Hair Care Products (non-coloring)").all())

    def test_discontinued_date_range_is_half_open(self):
        results = queries.find_discontinued_between(date(2020, 1, 1), date(2021, 1, 1))
        self.assertGreater(len(results), 0)
        self.assertTrue(results["DiscontinuedDate"].between("2020-01-01", "2020-12-31").all())

    def test_trends_return_distinct_product_counts(self):
        results = queries.reporting_trends(company_name="New Avon LLC")
        self.assertGreater(len(results), 0)
        self.assertEqual(list(results.columns), ["year", "ingredient_records", "product_count"])
        self.assertTrue((results["product_count"] <= results["ingredient_records"]).all())

    def test_rejects_unsupported_date_fields(self):
        with self.assertRaises(ValueError):
            queries.find_by_date("ReportingDate", "2020-01-01", "2021-01-01")

    def test_grouped_product_counts_are_distinct_ordered_and_limited(self):
        groups, sql = queries.grouped_product_counts(group_by="chemical", limit=10)
        self.assertEqual(len(groups), 10)
        self.assertEqual(groups.iloc[0]["group_value"], "Titanium dioxide")
        self.assertEqual(groups.iloc[0]["product_count"], 32010)
        self.assertEqual(int(groups.iloc[0]["total_groups"]), 123)
        self.assertTrue(groups["product_count"].is_monotonic_decreasing)
        self.assertIn("COUNT(DISTINCT CDPHId)", sql)
        self.assertIn("GROUP BY TRIM(ChemicalName)", sql)
        self.assertIn("ORDER BY product_count DESC", sql)
        self.assertIn("LIMIT ?", sql)
        self.assertNotIn("LIMIT 10", sql)

    def test_grouped_product_counts_apply_bound_filters_and_allow_ascending_order(self):
        companies, sql = queries.grouped_product_counts(
            group_by="company",
            chemical_name="Titanium dioxide",
            order_direction="asc",
            limit=3,
        )
        self.assertIn("?", sql)
        self.assertNotIn("Titanium dioxide", sql)
        self.assertIn("ORDER BY product_count ASC", sql)
        self.assertTrue(companies["product_count"].is_monotonic_increasing)

    def test_grouped_product_counts_reject_unapproved_dimensions(self):
        with self.assertRaises(ValueError):
            queries.grouped_product_counts(group_by="ProductCategory")


if __name__ == "__main__":
    unittest.main()