from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import duckdb

from src.config.settings import DATABASE_PATH


DB_FILE = DATABASE_PATH
EVIDENCE_COLUMNS = (
    "CDPHId", "CSFId", "CompanyId", "ChemicalId", "CasId", "PrimaryCategoryId",
    "SubCategoryId", "ProductName", "CompanyName", "BrandName", "PrimaryCategory",
    "SubCategory", "ChemicalName", "CasNumber", "InitialDateReported",
    "MostRecentDateReported", "DiscontinuedDate", "ChemicalDateRemoved", "ChemicalCount",
)
FILTER_COLUMNS = {
    "cas_number": "CasNumber",
    "chemical_name": "ChemicalName",
    "company_name": "CompanyName",
    "brand_name": "BrandName",
    "product_name": "ProductName",
    "primary_category": "PrimaryCategory",
    "subcategory": "SubCategory",
}
DATE_COLUMNS = {
    "reported": "MostRecentDateReported",
    "most_recent_reported": "MostRecentDateReported",
    "discontinued": "DiscontinuedDate",
    "removed": "ChemicalDateRemoved",
}
ENTITY_COLUMNS = {
    "company": "CompanyName",
    "brand": "BrandName",
    "product": "ProductName",
    "chemical": "ChemicalName",
    "cas": "CasNumber",
    "category": "PrimaryCategory",
    "subcategory": "SubCategory",
}


def _as_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _filters(
    *,
    cas_number: str | None = None,
    chemical_name: str | None = None,
    company_name: str | None = None,
    brand_name: str | None = None,
    product_name: str | None = None,
    primary_category: str | None = None,
    subcategory: str | None = None,
    date_field: str | None = None,
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    date_operator: str | None = None,
) -> tuple[list[str], list[Any]]:
    conditions: list[str] = []
    parameters: list[Any] = []
    values = {
        "cas_number": cas_number,
        "chemical_name": chemical_name,
        "company_name": company_name,
        "brand_name": brand_name,
        "product_name": product_name,
        "primary_category": primary_category,
        "subcategory": subcategory,
    }
    for filter_name, value in values.items():
        if value is not None:
            column = FILTER_COLUMNS[filter_name]
            conditions.append(f"LOWER(TRIM({column})) = LOWER(TRIM(?))")
            parameters.append(value)

    if date_field is not None:
        if date_field not in DATE_COLUMNS:
            supported = ", ".join(DATE_COLUMNS)
            raise ValueError(f"Unsupported date_field {date_field!r}; choose from: {supported}")
        column = DATE_COLUMNS[date_field]
        start = _as_date(date_from)
        end = _as_date(date_to)
        operator = date_operator or ("exists" if date_field == "discontinued" and start is None and end is None else "range")
        if operator == "exists":
            conditions.append(f"{column} IS NOT NULL")
        elif operator == "after":
            if start is None:
                raise ValueError("date_from is required for an after date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} > ?"])
            parameters.append(start)
        elif operator == "before":
            if start is None:
                raise ValueError("date_from is required for a before date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} < ?"])
            parameters.append(start)
        elif operator == "on":
            if start is None:
                raise ValueError("date_from is required for an on date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} >= ?", f"{column} < ?"])
            parameters.extend([start, start + timedelta(days=1)])
        elif operator == "from":
            if start is None:
                raise ValueError("date_from is required for a from date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} >= ?"])
            parameters.append(start)
        elif operator == "through":
            if start is None:
                raise ValueError("date_from is required for a through date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} <= ?"])
            parameters.append(start)
        elif operator == "between":
            if start is None or end is None:
                raise ValueError("date_from and date_to are required for a between date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} >= ?", f"{column} <= ?"])
            parameters.extend([start, end])
        elif operator == "after_before":
            if start is None or end is None:
                raise ValueError("date_from and date_to are required for an after/before date filter")
            conditions.extend([f"{column} IS NOT NULL", f"{column} > ?", f"{column} < ?"])
            parameters.extend([start, end])
        elif operator == "range":
            if start is None and end is None:
                raise ValueError("At least one of date_from or date_to is required with date_field")
            conditions.append(f"{column} IS NOT NULL")
            if start is not None:
                conditions.append(f"{column} >= ?")
                parameters.append(start)
            if end is not None:
                conditions.append(f"{column} < ?")
                parameters.append(end)
        else:
            raise ValueError(f"Unsupported date_operator {operator!r}")
    elif date_from is not None or date_to is not None:
        raise ValueError("date_field is required when a date range is provided")

    return conditions, parameters


def search_cosmetics(
    *,
    cas_number: str | None = None,
    chemical_name: str | None = None,
    company_name: str | None = None,
    brand_name: str | None = None,
    product_name: str | None = None,
    primary_category: str | None = None,
    subcategory: str | None = None,
    date_field: str | None = None,
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    date_operator: str | None = None,
    limit: int | None = None,
):
    """Return distinct ingredient records matching any combination of exact filters."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer or None")
    conditions, parameters = _filters(
        cas_number=cas_number,
        chemical_name=chemical_name,
        company_name=company_name,
        brand_name=brand_name,
        product_name=product_name,
        primary_category=primary_category,
        subcategory=subcategory,
        date_field=date_field,
        date_from=date_from,
        date_to=date_to,
        date_operator=date_operator,
    )
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    limit_clause = " LIMIT ?" if limit is not None else ""
    if limit is not None:
        parameters.append(limit)
    query = (
        f"SELECT DISTINCT {', '.join(EVIDENCE_COLUMNS)} FROM cosmetics{where_clause} "
        f"ORDER BY CDPHId, ChemicalId{limit_clause}"
    )
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        return connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()


def find_by_cas(cas_number: str):
    return search_cosmetics(cas_number=cas_number)


def find_by_chemical(chemical_name: str):
    return search_cosmetics(chemical_name=chemical_name)


def find_by_company(company_name: str):
    return search_cosmetics(company_name=company_name)


def find_by_brand(brand_name: str):
    return search_cosmetics(brand_name=brand_name)


def find_by_product(product_name: str):
    return search_cosmetics(product_name=product_name)


def find_by_category(primary_category: str, subcategory: str | None = None):
    return search_cosmetics(primary_category=primary_category, subcategory=subcategory)


def find_by_date(date_field: str, date_from: date | str, date_to: date | str):
    return search_cosmetics(date_field=date_field, date_from=date_from, date_to=date_to, date_operator="range")


def find_discontinued_between(date_from: date | str, date_to: date | str):
    return find_by_date("discontinued", date_from, date_to)


def reporting_trends(
    *,
    company_name: str | None = None,
    brand_name: str | None = None,
    primary_category: str | None = None,
    subcategory: str | None = None,
    date_field: str = "reported",
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    date_operator: str | None = None,
):
    """Aggregate ingredient-record and distinct-product counts by year."""
    if date_field not in DATE_COLUMNS:
        supported = ", ".join(DATE_COLUMNS)
        raise ValueError(f"Unsupported date_field {date_field!r}; choose from: {supported}")
    date_column = DATE_COLUMNS[date_field]
    conditions, parameters = _filters(
        company_name=company_name,
        brand_name=brand_name,
        primary_category=primary_category,
        subcategory=subcategory,
        date_field=date_field if date_from is not None or date_to is not None else None,
        date_from=date_from,
        date_to=date_to,
        date_operator=date_operator,
    )
    conditions.append(f"{date_column} IS NOT NULL")
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT EXTRACT(YEAR FROM {date_column})::INTEGER AS year,
               COUNT(*) AS ingredient_records,
               COUNT(DISTINCT CDPHId) AS product_count
        FROM cosmetics
        WHERE {where_clause}
        GROUP BY year
        ORDER BY year
    """
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        return connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()


def entity_values(entity_type: str) -> list[str]:
    if entity_type not in ENTITY_COLUMNS:
        supported = ", ".join(ENTITY_COLUMNS)
        raise ValueError(f"Unsupported entity_type {entity_type!r}; choose from: {supported}")
    column = ENTITY_COLUMNS[entity_type]
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        rows = connection.execute(
            f"SELECT DISTINCT TRIM({column}) FROM cosmetics "
            f"WHERE {column} IS NOT NULL AND TRIM({column}) <> '' ORDER BY 1"
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def count_cosmetics(**filters: Any) -> dict[str, int]:
    conditions, parameters = _filters(**filters)
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    query = (
        "SELECT COUNT(*) AS ingredient_records, COUNT(DISTINCT CDPHId) AS product_count "
        f"FROM cosmetics{where_clause}"
    )
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        row = connection.execute(query, parameters).fetchone()
        return {"ingredient_records": row[0], "product_count": row[1]}
    finally:
        connection.close()


def chemical_breakdown(**filters: Any):
    conditions, parameters = _filters(**filters)
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT TRIM(ChemicalName) AS ChemicalName, TRIM(CasNumber) AS CasNumber,
               COUNT(*) AS ingredient_records,
               COUNT(DISTINCT CDPHId) AS product_count
        FROM cosmetics{where_clause}
        GROUP BY TRIM(ChemicalName), TRIM(CasNumber)
        ORDER BY ChemicalName, CasNumber
    """
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        return connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()


def company_breakdown(**filters: Any):
    conditions, parameters = _filters(**filters)
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT TRIM(CompanyName) AS CompanyName,
               COUNT(DISTINCT CDPHId) AS product_count,
               COUNT(*) AS ingredient_records
        FROM cosmetics{where_clause}
        WHERE CompanyName IS NOT NULL AND TRIM(CompanyName) <> ''
        GROUP BY TRIM(CompanyName)
        ORDER BY CompanyName
    """
    if where_clause:
        query = query.replace(f"FROM cosmetics{where_clause}\n        WHERE", f"FROM cosmetics{where_clause} AND")
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        return connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()


def dataset_statistics() -> dict[str, Any]:
    query = """
        SELECT COUNT(*) AS ingredient_records,
               COUNT(DISTINCT CDPHId) AS product_count,
               MIN(InitialDateReported) AS first_reported,
               MAX(MostRecentDateReported) AS latest_reported,
               COUNT(*) FILTER (WHERE DiscontinuedDate IS NOT NULL) AS discontinued_records
        FROM cosmetics
    """
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        row = connection.execute(query).fetchone()
        return dict(zip(
            ("ingredient_records", "product_count", "first_reported", "latest_reported", "discontinued_records"),
            row,
        ))
    finally:
        connection.close()


def date_coverage(date_field: str, **filters: Any) -> dict[str, Any]:
    """Summarize observed lifecycle date coverage for the given non-date filters."""
    if date_field not in DATE_COLUMNS:
        supported = ", ".join(DATE_COLUMNS)
        raise ValueError(f"Unsupported date_field {date_field!r}; choose from: {supported}")
    conditions, parameters = _filters(**filters)
    column = DATE_COLUMNS[date_field]
    conditions.append(f"{column} IS NOT NULL")
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT COUNT(*) AS ingredient_records,
               COUNT(DISTINCT CDPHId) AS product_count,
               MIN({column}) AS first_date,
               MAX({column}) AS latest_date
        FROM cosmetics
        WHERE {where_clause}
    """
    connection = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        row = connection.execute(query, parameters).fetchone()
        return dict(zip(("ingredient_records", "product_count", "first_date", "latest_date"), row))
    finally:
        connection.close()


__all__ = [
    "DB_FILE", "search_cosmetics", "find_by_cas", "find_by_chemical", "find_by_company",
    "find_by_brand", "find_by_product", "find_by_category", "find_by_date",
    "find_discontinued_between", "reporting_trends", "entity_values", "count_cosmetics",
    "chemical_breakdown", "dataset_statistics",
    "company_breakdown",
]