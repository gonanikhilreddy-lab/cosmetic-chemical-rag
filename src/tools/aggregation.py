from typing import Any

from src.tools.structured_query import count_cosmetics, dataset_statistics, reporting_trends


def summarize_reporting_trends(**filters: Any):
    return reporting_trends(**filters)


def summarize_dataset() -> dict[str, Any]:
    return dataset_statistics()


def count_products_and_records(**filters: Any) -> dict[str, int]:
    return count_cosmetics(**filters)