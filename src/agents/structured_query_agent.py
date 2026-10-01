import re
from typing import Any

import pandas as pd

from src.tools.structured_query import (
    chemical_breakdown,
    company_breakdown,
    count_cosmetics,
    date_predicate,
    date_coverage,
    dataset_statistics,
    grouped_product_counts,
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
    def retrieve(self, question: str, intent: str, entities: dict[str, str], comparisons: dict[str, list[str]], date_field: str | None, date_from: str | None, date_to: str | None, date_operator: str | None, output_targets: list[str], require_discontinued: bool, limit: int, group_by: str | None = None, order_by: str | None = None, order_direction: str = "desc", top_n: int | None = None) -> dict[str, Any]:
        if intent == "data_quality":
            return {"counts": {}, "aggregate": {"dataset": dataset_statistics()}, "evidence": []}
        filters = {ENTITY_TO_FILTER[kind]: value for kind, value in entities.items() if kind in ENTITY_TO_FILTER}
        if date_field and (date_from or date_to or date_operator == "exists"):
            filters.update({"date_field": date_field, "date_from": date_from, "date_to": date_to, "date_operator": date_operator})
        if require_discontinued and date_field != "discontinued":
            filters["require_discontinued"] = True
        sql_date_predicate = date_predicate(date_field, date_from, date_to, date_operator)

        if intent == "aggregation" and group_by:
            effective_limit = top_n or limit
            order_column = order_by or "product_count"
            groups, aggregation_sql = grouped_product_counts(
                group_by=group_by,
                order_by=order_column,
                order_direction=order_direction,
                limit=effective_limit,
                **filters,
            )
            total_group_count = int(groups["total_groups"].iloc[0]) if not groups.empty else 0
            group_records = _records(groups.drop(columns=["total_groups"]))
            return {
                "counts": {},
                "aggregate": {
                    "groups": group_records,
                    "group_by": group_by,
                    "group_count": total_group_count,
                    "returned_count": len(groups),
                    "measure": order_column,
                    "order_direction": order_direction,
                    "limit": effective_limit,
                },
                "evidence": [],
                "filters": filters,
                "result_type": "aggregation",
                "aggregation_sql": aggregation_sql,
                "aggregation_limit": effective_limit,
                "sql_date_predicate": sql_date_predicate,
            }

        if "product_count" in output_targets and ("company_list" in output_targets or "company_count" in output_targets):
            company_rows = company_breakdown(**filters)
            return {
                "counts": count_cosmetics(**filters),
                "aggregate": {
                    "companies": _records(company_rows.head(limit)),
                    "company_count": len(company_rows),
                },
                "evidence": [],
                "filters": filters,
                "result_type": "multi",
                "sql_date_predicate": sql_date_predicate,
            }

        if intent in ("company_lookup", "company_count", "company_aggregation"):
            company_rows = company_breakdown(**filters)
            companies = _records(company_rows.head(limit))
            aggregate = {"companies": companies, "company_count": len(company_rows)}
            return {
                "counts": count_cosmetics(**filters),
                "aggregate": aggregate,
                "evidence": [],
                "filters": filters,
                "result_type": "companies",
                "sql_date_predicate": sql_date_predicate,
            }

        if intent == "chemical_count":
            chemical_rows = chemical_breakdown(**filters)
            return {
                "counts": count_cosmetics(**filters),
                "aggregate": {
                    "chemicals": _records(chemical_rows.head(limit)),
                    "chemical_count": len(chemical_rows),
                },
                "evidence": [],
                "filters": filters,
                "result_type": "chemicals",
                "sql_date_predicate": sql_date_predicate,
            }

        if intent == "product_count":
            return {
                "counts": count_cosmetics(**filters),
                "aggregate": {},
                "evidence": [],
                "filters": filters,
                "result_type": "products",
                "sql_date_predicate": sql_date_predicate,
            }

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
                if key not in ("date_field", "date_from", "date_to", "date_operator")
            }
            aggregate["date_coverage"] = date_coverage(date_field, **non_date_filters)
        if intent == "trend":
            trend_filters = {key: value for key, value in filters.items() if key not in ("date_field", "date_from", "date_to", "date_operator")}
            trend = reporting_trends(
                **trend_filters,
                date_field=date_field or "reported",
                date_from=date_from,
                date_to=date_to,
                date_operator=date_operator,
            )
            aggregate["trend"] = _records(trend)
        if intent == "summarize" or re.search(r"what chemicals|chemicals reported|which chemicals", question, re.IGNORECASE):
            aggregate["chemicals"] = _records(chemical_breakdown(**filters).head(limit))
        return {"counts": counts, "aggregate": aggregate, "evidence": evidence, "filters": filters, "result_type": "products", "sql_date_predicate": sql_date_predicate}