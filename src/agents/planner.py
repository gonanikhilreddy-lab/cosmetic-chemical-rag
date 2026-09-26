import re
from datetime import datetime

from src.schemas.query_plan import QueryPlan


DATE_PHRASES = (
    ("most recent report", "most_recent_reported"),
    ("discontinued", "discontinued"),
    ("removed", "removed"),
    ("reformulated", "removed"),
    ("reported", "reported"),
)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
EXPLICIT_DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
SAFETY_TERMS = (
    "safe",
    "safety",
    "pregnancy",
    "pregnant",
    "side effect",
    "toxic",
    "harmful",
    "health effect",
    "cancer",
    "allergy",
    "safe for children",
    "breastfeeding",
)


class PlannerAgent:
    def plan(self, question: str) -> QueryPlan:
        text = question.lower()
        if any(term in text for term in SAFETY_TERMS):
            intent = "safety_question"
        elif re.search(r"\bhow many companies\b", text):
            intent = "company_count"
        elif re.search(r"\b(?:which|what)\s+companies\b|\bcompanies\s+(?:that|reported)\b", text):
            intent = "company_aggregation" if "how many products" in text or "product count" in text else "company_lookup"
        elif any(term in text for term in ("data quality", "missing values", "dataset stats", "how many rows", "dataset size")):
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
        date_operator = None
        date_from = None
        date_to = None
        explicit_dates = [
            datetime.strptime(match.group(0), "%B %d, %Y").date().isoformat()
            for match in EXPLICIT_DATE_PATTERN.finditer(question)
        ]
        if date_field and explicit_dates:
            if re.search(r"\bafter\b", text) and re.search(r"\bbefore\b", text) and len(explicit_dates) >= 2:
                date_operator = "after_before"
                date_from, date_to = explicit_dates[:2]
            elif re.search(r"\bbetween\b", text) and len(explicit_dates) >= 2:
                date_operator = "between"
                date_from, date_to = explicit_dates[:2]
            elif re.search(r"\bafter\b", text):
                date_operator = "after"
                date_from = explicit_dates[0]
            elif re.search(r"\bbefore\b", text):
                date_operator = "before"
                date_from = explicit_dates[0]
            elif re.search(r"\bon\b", text):
                date_operator = "on"
                date_from = explicit_dates[0]
            elif re.search(r"\bthrough\b|\buntil\b", text):
                date_operator = "through"
                date_from = explicit_dates[0]
            else:
                date_operator = "from"
                date_from = explicit_dates[0]
        else:
            years = [int(value) for value in YEAR_PATTERN.findall(text)]
            if date_field and years:
                date_operator = "range"
                date_from = f"{min(years)}-01-01"
                date_to = f"{max(years) + 1}-01-01"
        if date_field == "discontinued" and date_operator is None:
            date_operator = "exists"
        warnings = []
        years = [int(value) for value in YEAR_PATTERN.findall(text)]
        if years and not date_field:
            warnings.append("A year was mentioned without a lifecycle date; no date filter was applied.")
        return QueryPlan(
            intent=intent,
            date_field=date_field,
            date_from=date_from,
            date_to=date_to,
            date_operator=date_operator,
            retrieval_mode="structured",
            warnings=warnings,
        )