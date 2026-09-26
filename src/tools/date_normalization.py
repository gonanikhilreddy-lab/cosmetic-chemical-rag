from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re


YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
EXPLICIT_DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedDate:
    operator: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    raw_expression: str | None = None


def normalize_date_expression(question: str, date_field: str | None) -> NormalizedDate:
    if date_field is None:
        return NormalizedDate()

    text = question.lower()
    explicit_dates = [
        datetime.strptime(match.group(0), "%B %d, %Y").date()
        for match in EXPLICIT_DATE_PATTERN.finditer(question)
    ]
    years = [int(value) for value in YEAR_PATTERN.findall(text)]
    raw_expression = " ".join(question.split())

    if len(explicit_dates) >= 2 and re.search(r"\bbetween\b", text):
        return NormalizedDate(
            "range",
            explicit_dates[0].isoformat(),
            (explicit_dates[1] + timedelta(days=1)).isoformat(),
            raw_expression,
        )
    if len(years) >= 2 and (
        re.search(r"\bbetween\b", text)
        or re.search(r"\bfrom\b.*\b(?:to|through)\b", text)
    ):
        return NormalizedDate(
            "range",
            f"{min(years)}-01-01",
            f"{max(years) + 1}-01-01",
            raw_expression,
        )
    if explicit_dates:
        current = explicit_dates[0]
        if re.search(r"\bafter\b", text):
            operator = "after"
        elif re.search(r"\bbefore\b", text):
            operator = "before"
        elif re.search(r"\bon\b", text):
            operator = "on"
        elif re.search(r"\bthrough\b", text):
            operator = "through"
        elif re.search(r"\buntil\b", text):
            operator = "range"
            return NormalizedDate(operator, None, (current + timedelta(days=1)).isoformat(), raw_expression)
        else:
            operator = "from"
        return NormalizedDate(operator, current.isoformat(), None, raw_expression)
    if years:
        year = max(years)
        if re.search(r"\bafter\b", text):
            return NormalizedDate("after", f"{year + 1}-01-01", None, raw_expression)
        if re.search(r"\bbefore\b", text):
            return NormalizedDate("before", f"{year}-01-01", None, raw_expression)
        if re.search(r"\buntil\b", text):
            return NormalizedDate("range", None, f"{year + 1}-01-01", raw_expression)
        if re.search(r"\bfrom\b", text):
            return NormalizedDate("from", f"{year}-01-01", None, raw_expression)
        return NormalizedDate("range", f"{min(years)}-01-01", f"{max(years) + 1}-01-01", raw_expression)
    return NormalizedDate(None, None, None, raw_expression)
