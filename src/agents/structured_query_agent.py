import re
from typing import Any

import pandas as pd

from src.tools.structured_query import (
    chemical_breakdown,
    count_cosmetics,
    date_coverage,
    dataset_statistics,
    search_cosmetics,
    reporting_trends,
)


ENTITY_TO_FILTER = {
    "company": "company_name",
    "brand": "brand_name",
    "product": "product_name",
    "chemical": "chemical_name",
    "cas": "cas_number",
    "category": "primary_category",
    "subcategory": "subcategory",
}


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


class StructuredQueryAgent:
    def retrieve(self, question: str, intent: str, entities: dict[str, str], comparisons: dict[str, list[str]], date_field: str | None, date_from: str | None, date_to: str | None, limit: int) -> dict[str, Any]:
        if intent == "data_quality":
            return {"counts": {}, "aggregate": {"dataset": dataset_statistics()}, "evidence": []}
        filters = {ENTITY_TO_FILTER[kind]: value for kind, value in entities.items() if kind in ENTITY_TO_FILTER}
        if date_field and date_from:
            filters.update({"date_field": date_field, "date_from": date_from, "date_to": date_to})

        if intent == "compare" and comparisons:
            kind, values = next(iter(comparisons.items()))
            filter_name = ENTITY_TO_FILTER[kind]
            comparison = []
            evidence = []
            for value in values:
                selected = {**filters, filter_name: value}
                summary = count_cosmetics(**selected)
                records = _records(search_cosmetics(**selected, limit=min(limit, 5)))
                comparison.append({"entity_type": kind, "entity": value, **summary, "evidence": records})
                evidence.extend(records)
            return {"counts": {}, "aggregate": {"comparison": comparison}, "evidence": evidence[:limit], "filters": filters}

        counts = count_cosmetics(**filters)
        evidence = _records(search_cosmetics(**filters, limit=limit))
        aggregate: dict[str, Any] = {}
        if counts["ingredient_records"] == 0 and date_field:
            non_date_filters = {
                key: value for key, value in filters.items()
                if key not in ("date_field", "date_from", "date_to")
            }
            aggregate["date_coverage"] = date_coverage(date_field, **non_date_filters)
        if intent == "trend":
            trend_filters = {key: value for key, value in filters.items() if key not in ("date_field", "date_from", "date_to")}
            trend = reporting_trends(
                **trend_filters,
                date_field=date_field or "reported",
                date_from=date_from,
                date_to=date_to,
            )
            aggregate["trend"] = _records(trend)
        if intent == "summarize" or re.search(r"what chemicals|chemicals reported|which chemicals", question, re.IGNORECASE):
            aggregate["chemicals"] = _records(chemical_breakdown(**filters).head(limit))
        return {"counts": counts, "aggregate": aggregate, "evidence": evidence, "filters": filters}