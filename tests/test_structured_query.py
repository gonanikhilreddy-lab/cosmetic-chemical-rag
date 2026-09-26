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


if __name__ == "__main__":
    unittest.main()