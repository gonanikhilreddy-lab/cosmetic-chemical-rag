import re

from src.schemas.query_plan import QueryPlan


DATE_PHRASES = (
    ("most recent report", "most_recent_reported"),
    ("discontinued", "discontinued"),
    ("removed", "removed"),
    ("reformulated", "removed"),
    ("reported", "reported"),
)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")


class PlannerAgent:
    def plan(self, question: str) -> QueryPlan:
        text = question.lower()
        if any(term in text for term in ("data quality", "missing values", "dataset stats", "how many rows", "dataset size")):
            intent = "data_quality"
        elif any(term in text for term in ("compare", "comparison", "versus", " vs ")):
            intent = "compare"
        elif any(term in text for term in ("trend", "over time", "by year", "yearly")):
            intent = "trend"
        elif any(term in text for term in ("summarize", "summary", "overview")):
            intent = "summarize"
        elif any(term in text for term in ("list", "show", "which products")):
            intent = "list"
        else:
            intent = "lookup"

        date_field = next((field for phrase, field in DATE_PHRASES if phrase in text), None)
        if intent == "trend" and date_field is None:
            date_field = "reported"
        years = [int(value) for value in YEAR_PATTERN.findall(text)]
        date_from = f"{min(years)}-01-01" if date_field and years else None
        date_to = f"{max(years) + 1}-01-01" if date_field and years else None
        warnings = []
        if years and not date_field:
            warnings.append("A year was mentioned without a lifecycle date; no date filter was applied.")
        return QueryPlan(
            intent=intent,
            date_field=date_field,
            date_from=date_from,
            date_to=date_to,
            retrieval_mode="structured",
            warnings=warnings,
        )